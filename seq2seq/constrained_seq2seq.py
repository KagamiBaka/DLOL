#!/usr/bin/env python
# -*- coding:utf-8 -*-
import logging
import os

import torch
import torch.nn as nn
from transformers import (
    PreTrainedTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import torch.nn.functional as F

from torch.cuda.amp import autocast


from torch.utils.tensorboard import SummaryWriter
from extraction.event_schema import EventSchema
from extraction.extract_constraint import get_constraint_decoder
from extraction.extraction_metrics import get_extract_metrics
from seq2seq.label_smoother_sum import SumLabelSmoother
from seq2seq.utils import lmap
import itertools
import math
import collections
from dataclasses import dataclass, field
from typing import Union, List, Callable, Dict, Tuple, Any, Optional
import numpy as np
import random
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    get_full_repo_name,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)
logger = logging.getLogger(__name__)

if is_sagemaker_mp_enabled():
    import transformers.smdistributed.modelparallel.torch as smp

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
def add_logging_file(training_args):
    fh = logging.FileHandler(os.path.join(training_args.output_dir.rstrip(os.sep) + '.log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)


def decode_tree_str(sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor"],
                    tokenizer: PreTrainedTokenizer) -> List[str]:
    def clean_tree_text(x):
        return x.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()

    sequences = np.where(sequences != -100, sequences, tokenizer.pad_token_id)

    str_list = tokenizer.batch_decode(sequences, skip_special_tokens=False)
    return lmap(clean_tree_text, str_list)


def build_compute_extract_metrics_event_fn(decoding_type_schema: EventSchema,
                                           decoding_format: str,
                                           tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        return decode_tree_str(pred.predictions, tokenizer), decode_tree_str(pred.label_ids, tokenizer)

    def extraction_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        extraction = get_extract_metrics(pred_lns=pred_str, tgt_lns=label_str, label_constraint=decoding_type_schema,
                                         decoding_format=decoding_format)
        # rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        extraction.update({"gen_len": summ_len})
        # extraction.update( )
        return extraction

    compute_metrics_fn = extraction_metrics
    return compute_metrics_fn

class Seq2SetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tgt_tokens, pred, branch_nums):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        # loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        ce_losses = torch.zeros(tgt_tokens.shape[0]).to('cuda')
        seq_probs = torch.zeros(tgt_tokens.shape[0]).to('cuda')
        branch_loss = torch.zeros(len(branch_nums)).to('cuda')

        for i in range(tgt_tokens.shape[0]):
            ce_losses[i] = F.cross_entropy(target=tgt_tokens[i:i+1], input=pred.transpose(1, 2)[i:i+1])
            seq_probs[i] = - sum([0 if  y == -100 else 1 for y in tgt_tokens[i]]) * ce_losses[i]
        
        out_seq_probs = []
        start = 0
        tmp = torch.exp(seq_probs).detach().cpu().numpy()
        for i, branch_num in enumerate(branch_nums):
            out_seq_probs.append(tmp[start: start + branch_num])
            start += branch_num

        start = 0
        for i, branch_num in enumerate(branch_nums):
            branch_loss[i] = - torch.log(torch.mean(torch.exp(-ce_losses[start: start + branch_num])))
            start += branch_num

        loss = torch.mean(branch_loss)
        return loss, out_seq_probs

@dataclass
class ConstraintSeq2SeqTrainingArguments(Seq2SeqTrainingArguments):
    """
    Parameters:
        constraint_decoding (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use Constraint Decoding
        structure_weight (:obj:`float`, `optional`, defaults to :obj:`None`):
    """
    constraint_decoding: bool = field(default=False, metadata={"help": "Whether to Constraint Decoding or not."})
    random_order: bool = field(default=False, metadata={"help": ""})
    vanilla_seq2seq: bool = field(default=False, metadata={"help": ""})
    label_smoothing_sum: bool = field(default=False,
                                      metadata={"help": "Whether to use sum token loss for label smoothing"})
    save_steps: int = field(default=500,
                                      metadata={"help": ""})
    is_fp16: int = field(default=0,
                                      metadata={"help": ""})
    accumulate_steps: int = field(default=1,
                                      metadata={"help": ""})
    _dataset_name: str = field(default='reuters',
                                      metadata={"help": "dataset_name"})
    _loss_type: str = field(default='set',
                                      metadata={"help": "loss type"})


class ConstraintSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, decoding_type_schema=None, decoding_format='tree', source_prefix=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = 0
        self._max_length=512
        self._num_beams=4
        self.decoding_format = decoding_format
        self.decoding_type_schema = decoding_type_schema
        self.losser = Seq2SetLoss()
        self.pruning_round = 0
        self.step = 0
        self.update_step = 0
        self.end_furcate = 0
        self.current_trees = {} # dict, input_tokens: 
        # Label smoothing by sum token loss, different from different Label smootheing
        if self.args.label_smoothing_sum and self.args.label_smoothing_factor != 0:
            print('loss_type: ', self.args._loss_type)
            self.label_smoother = SumLabelSmoother(loss_type= self.args._loss_type, epsilon=self.args.label_smoothing_factor)
            print('Using %s' % self.label_smoother)
        elif self.args.label_smoothing_factor != 0:
            print('Using %s' % self.label_smoother)
        else:
            self.label_smoother = None

        if self.args.constraint_decoding:
            self.constraint_decoder = get_constraint_decoder(tokenizer=self.tokenizer,
                                                             type_schema=self.decoding_type_schema,
                                                             decoding_schema=self.decoding_format,
                                                             source_prefix=source_prefix)
        else:
            self.constraint_decoder = None

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        def prefix_allowed_tokens_fn(batch_id, sent):
            # print(self.tokenizer.convert_ids_to_tokens(inputs['labels'][batch_id]))
            src_sentence = inputs['input_ids'][batch_id]
            return self.constraint_decoder.constraint_decoding(src_sentence=src_sentence,
                                                               tgt_generated=sent)

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model=model,
                inputs=inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn if self.constraint_decoder else None,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        gen_kwargs = {
            "max_length": self._max_length if self._max_length is not None else self.model.config.max_length,
            "num_beams": self._num_beams if self._num_beams is not None else self.model.config.num_beams,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn if self.constraint_decoder else None,
        }

        generated_tokens = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **gen_kwargs,
        )

        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_kwargs["max_length"])

        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if has_labels:
                loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        labels = inputs["labels"]
        if labels.shape[-1] < gen_kwargs["max_length"]:
            labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])

        return loss, generated_tokens, labels


    def convert_labels_to_current_tree_path_for_training(self, labels, current_tree):
        def get_all_nodes(labels):
            nodes = []
            tmp = []
            for i in labels[1:]: # skip start token '0'
                if i != 6 and i != 2:
                    tmp.append(i)
                else:
                    nodes.append(tmp)
                    tmp = []
            return nodes
        
        def get_remaining_nodes(nodes, current_tree):
            remaining_nodes = []
            for i, node in enumerate(nodes):
                if i not in current_tree:
                    remaining_nodes.append(node)
            return remaining_nodes
        
        def get_current_path(nodes, current_tree):
            # print(nodes, current_tree)
            path = [0]
            for i in current_tree:
                path += nodes[i] + [6]
            return path
        labels = list(labels.detach().cpu().numpy())
        # print(labels)
        nodes = get_all_nodes(labels) # nodes: [[..], [..], [..]...]

        current_path = get_current_path(nodes, current_tree)
        
        remaining_nodes = get_remaining_nodes(nodes, current_tree)


        if remaining_nodes == []:
            branchs = [current_path + [2]]
        else:
            branchs = []
            if len(remaining_nodes) > 1:
                for node in remaining_nodes:
                    # branch = current_path + node + [2] # ablation
                    branch = current_path + node + [734] # ablation # 734在bart中是...
                    branchs.append(branch)
            else:
                for node in remaining_nodes:
                    branch = current_path + node + [2]
                    branchs.append(branch)
        # print(branchs)
        return branchs

    def convert_labels_to_current_tree_path_for_pruning(self, labels, current_tree):
        def get_all_nodes(labels):
            nodes = []
            tmp = []
            for i in labels[1:]: # skip start token '0'
                if i != 6 and i != 2:
                    tmp.append(i)
                else:
                    nodes.append(tmp)
                    tmp = []
            return nodes
        
        def get_remaining_nodes(nodes, current_tree):
            remaining_nodes = []
            for i, node in enumerate(nodes):
                if i not in current_tree:
                    remaining_nodes.append(node)
            return remaining_nodes
        
        def get_current_path(nodes, current_tree):
            # print(nodes, current_tree)
            path = [0]
            for i in current_tree:
                path += nodes[i] + [6]
            return path
        labels = list(labels.detach().cpu().numpy())
        # print(labels)
        nodes = get_all_nodes(labels) # nodes: [[..], [..], [..]...]

        current_path = get_current_path(nodes, current_tree)
        
        remaining_nodes = get_remaining_nodes(nodes, current_tree)
        # print(nodes, current_path, current_tree)
        
        if remaining_nodes == []:
            branchs = [current_path + [2]]
        else:
            branchs = []
            if len(remaining_nodes) > 1:
                for node in remaining_nodes:
                    # branch = current_path + node + [2] # ablation
                    branch = current_path + node + [734] # ablation # 734在bart中是"..."
                    branchs.append(branch)
            else:
                for node in remaining_nodes:
                    branch = current_path + node + [2]
                    branchs.append(branch)
        if len(branchs) == 1:
            return []
        return branchs
    
    def get_current_inputs_for_training(self, inputs):
        def drop_pad(ids):
            for i in range(len(ids)):
                if ids[i] == 2:
                    break
            return ids[0: i+1]
        
        def pad_labels(labels):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in labels:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [-100 for _ in range(pad_num)])
            return new_labels
        
        def pad_decoder_input_ids(decoder_input_ids):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in decoder_input_ids:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [1 for _ in range(pad_num)])
            return new_labels

        deleted_samples = [] # for
        branch_nums = []
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        labels = []
        new_inputs = {}
        for i in range(len(inputs["input_ids"])):
            dict_key = tuple(drop_pad(inputs["input_ids"][i].detach().cpu().numpy()))
            if dict_key not in self.current_trees.keys():
                # print(dict_key)
                # print(self.current_trees.keys())
                self.current_trees[dict_key] = []
                # print(self.current_trees)
                # print(self.current_trees[dict_key])
            # print(inputs["input_ids"][i], inputs["labels"][i])
            try:
                branchs = self.convert_labels_to_current_tree_path_for_training(inputs["labels"][i], self.current_trees[dict_key])
            except:
                deleted_samples.append(i)
                continue
            for branch in branchs:
                # print(branch)
                labels.append(branch)
                decoder_input_id = [2] + branch[:-1]
                decoder_input_ids.append(decoder_input_id)
            branch_nums += [len(branchs)]
            for _ in range(len(branchs)):
                input_ids.append(inputs["input_ids"][i])
                attention_mask.append(inputs["attention_mask"][i])

        new_inputs['input_ids'] = torch.stack(input_ids)
        new_inputs['attention_mask'] = torch.stack(attention_mask)
        # for i in range(len(decoder_input_ids)):
        #     print(decoder_input_ids[i])
        #     print(labels[i])
        decoder_input_ids = pad_decoder_input_ids(decoder_input_ids)
        # print(decoder_input_ids)
        new_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids).to('cuda')

        labels = pad_labels(labels)
        new_inputs['labels'] = torch.tensor(labels).to('cuda')
        return new_inputs, branch_nums, deleted_samples

    def get_current_inputs_for_pruning(self, inputs):
        def drop_pad(ids):
            for i in range(len(ids)):
                if ids[i] == 2:
                    break
            return ids[0: i+1]
        
        def pad_labels(labels):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in labels:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [-100 for _ in range(pad_num)])
            return new_labels
        
        def pad_decoder_input_ids(decoder_input_ids):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in decoder_input_ids:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [1 for _ in range(pad_num)])
            return new_labels

        deleted_samples = [] # for
        branch_nums = []
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        labels = []
        new_inputs = {}
        for i in range(len(inputs["input_ids"])):
            dict_key = tuple(drop_pad(inputs["input_ids"][i].detach().cpu().numpy()))
            if dict_key not in self.current_trees.keys():
                # print(dict_key)
                # print(self.current_trees.keys())
                self.current_trees[dict_key] = []
                # print(self.current_trees)
                # print(self.current_trees[dict_key])
            # print(inputs["input_ids"][i], inputs["labels"][i])
            try:
                branchs = self.convert_labels_to_current_tree_path_for_pruning(inputs["labels"][i], self.current_trees[dict_key])
            except:
                deleted_samples.append(i)
                continue
            for branch in branchs:
                # print(branch)
                labels.append(branch)
                decoder_input_id = [2] + branch[:-1]
                decoder_input_ids.append(decoder_input_id)
            branch_nums += [len(branchs)]
            for _ in range(len(branchs)):
                input_ids.append(inputs["input_ids"][i])
                attention_mask.append(inputs["attention_mask"][i])
        if len(labels) == 0:
            return None, None, None
        
        new_inputs['input_ids'] = torch.stack(input_ids)
        new_inputs['attention_mask'] = torch.stack(attention_mask)
        # for i in range(len(decoder_input_ids)):
        #     print(decoder_input_ids[i])
        #     print(labels[i])
        decoder_input_ids = pad_decoder_input_ids(decoder_input_ids)
        # print(decoder_input_ids)
        new_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids).to('cuda')

        labels = pad_labels(labels)
        new_inputs['labels'] = torch.tensor(labels).to('cuda')
        return new_inputs, branch_nums, deleted_samples
    
    def update_current_trees(self, input_id, prob):
        kept_branch_num = 1
        if len(prob) <= kept_branch_num: # three branch left, need not update
            return
        def drop_pad(ids):
            for i in range(len(ids)):
                if ids[i] == 2:
                    break
            return ids[0: i+1]
        input_id = tuple(drop_pad(input_id.detach().cpu().numpy()))
        path_len = len(self.current_trees[input_id]) + len(prob)

        _min = 0
        for i in range(len(prob)):
            # if prob[i] >= prob[_max]:
            if prob[i] <= prob[_min]: ### ablation
                _min = i
        
        tmp = []
        for i in range(path_len):
            if i not in self.current_trees[input_id]:
                tmp += [i]
        # print(self.current_trees, tmp, _max)
        # print(input_id, prob)
        # print(tmp[_max])
        self.current_trees[input_id] += [tmp[_min]]
    
    def get_random_order_inputs_for_training(self, inputs):
        def drop_pad(ids):
            for i in range(len(ids)):
                if ids[i] == 2:
                    break
            return ids[0: i+1]
        
        def pad_labels(labels):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in labels:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [-100 for _ in range(pad_num)])
            return new_labels
        
        def pad_decoder_input_ids(decoder_input_ids):
            max_len = max([len(i) for i in labels])
            new_labels = []
            for seq_label in decoder_input_ids:
                pad_num = max_len - len(seq_label)
                new_labels.append(seq_label + [1 for _ in range(pad_num)])
            return new_labels
        
        def get_all_nodes(labels):
            nodes = []
            tmp = []
            for i in labels[1:]: # skip start token '0'
                if i != 6 and i != 2:
                    tmp.append(i)
                else:
                    nodes.append(tmp)
                    tmp = []
            return nodes

        deleted_samples = [] # for
        branch_nums = []
        input_ids = []
        attention_mask = []
        decoder_input_ids = []
        labels = []
        new_inputs = {}
        for i in range(len(inputs["input_ids"])):
            nodes = get_all_nodes(inputs["labels"][i])
            random.shuffle(nodes)
            label = [0]
            for i in range(len(nodes)):
                label += nodes[i] + [6]
            label = label[:-1] + [2] + [-100]
            # if len(label) < len(inputs["labels"][i]):
            #     label += [-100 for i in range(len(inputs["labels"][i]) - len(label))]
            decoder_input_id = [2] + label[:-1]
            # if len(decoder_input_id) < len(inputs["decoder_input_ids"][i]):
            #     decoder_input_id += [-100 for i in range(len(inputs["decoder_input_ids"][i]) - len(decoder_input_id))]
            labels.append(label)
            decoder_input_ids.append(decoder_input_id)
            input_ids.append(inputs["input_ids"][i])
            attention_mask.append(inputs["attention_mask"][i])

        new_inputs['input_ids'] = inputs["input_ids"]
        new_inputs['attention_mask'] = inputs["attention_mask"]
        # for i in range(len(decoder_input_ids)):
        #     print(decoder_input_ids[i])
        #     print(labels[i])
        decoder_input_ids = pad_decoder_input_ids(decoder_input_ids)
        # print(decoder_input_ids)
        new_inputs['decoder_input_ids'] = torch.tensor(decoder_input_ids).to('cuda')

        labels = pad_labels(labels)
        new_inputs['labels'] = torch.tensor(labels).to('cuda')
        return new_inputs
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        vanilla_seq2seq = self.args.vanilla_seq2seq
        if self.args._dataset_name == 'GO-EMO':
            step_count = math.ceil(len(self.train_dataset) / self.args.train_batch_size)
            epoch_count = 5
        elif self.args._dataset_name == 'reuters':
            step_count =  math.ceil(len(self.train_dataset) / self.args.train_batch_size) # steps of one epoch
            epoch_count = 11 # order learning epochs, equals to the cardinal of dataset
        elif self.args._dataset_name == 'slashdot':
            step_count = math.ceil(len(self.train_dataset) / self.args.train_batch_size)
            epoch_count = 10

        # print("step_count: ", step_count)
            
        if not vanilla_seq2seq: # GO-EMO 5 Reuters 11 UFET 15 AAPD 8
            if self.step % step_count == 0 and self.step != 0 and self.pruning_round < epoch_count: 
                if self.update_step == step_count: # update step和一个epoch的step应该相同
                    self.update_step = 0
                    self.step += 1
                    self.pruning_round += 1

                else:
                    self.update_step += 1
                    model.eval()
                    inputs = self._prepare_inputs(inputs)
                    # print(inputs)
                    tmp_inputs, branch_nums, deleted_samples = self.get_current_inputs_for_pruning(inputs)
                    if tmp_inputs == None:
                        return torch.tensor(0.).to('cuda').detach()
                    loss, probs = self.compute_loss(model, tmp_inputs, branch_nums)
                    count = 0
                    for i in range(len(inputs['input_ids'])):
                        if i in deleted_samples:
                            # print(inputs['input_ids'][i], inputs['labels'][i])
                            continue
                        self.update_current_trees(inputs['input_ids'][i], probs[count])
                        count += 1
                    
                    return loss.detach()
            else:
                self.step += 1
        
        model.train()
        inputs = self._prepare_inputs(inputs)
        if not vanilla_seq2seq:
            inputs, branch_nums, _ = self.get_current_inputs_for_training(inputs)
        
        if self.args.random_order:
            inputs = self.get_random_order_inputs_for_training(inputs)

        # print(sum([len(i) for i in inputs['labels']]), sum([len(i) for i in inputs['input_ids']]), branch_nums)
        # print(len(inputs['labels']), branch_nums)
        if not vanilla_seq2seq:
            if self.use_amp:
                with autocast():
                    loss, _ = self.compute_loss(model, inputs, branch_nums)
            else:
                loss, _ = self.compute_loss(model, inputs, branch_nums)
        else:
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss_vanilla(model, inputs)
            else:
                loss = self.compute_loss_vanilla(model, inputs)
            

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        
            
        return loss.detach()
    
    def compute_loss_vanilla(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, branch_nums, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print(inputs)
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if self.label_smoother is not None:
            loss, probs = self.label_smoother(inputs.pop("labels"), outputs.logits, branch_nums)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss, probs = self.losser(inputs.pop("labels"), outputs.logits, branch_nums)

        return (loss, outputs) if return_outputs else loss, probs
        # loss = torch.mean(F.cross_entropy(target=inputs.pop("labels"), input=outputs.logits.transpose(1, 2)))
        # return loss, 0

    
    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.train_dataset, self.args.train_batch_size, model_input_name=model_input_name
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.train_dataset,
                    self.args.train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    model_input_name=model_input_name,
                )

        else:
            if self.args.world_size <= 1:
                # print('SequentialSampler')
                # ablation
                # return SequentialSampler(self.train_dataset)
                return RandomSampler(self.train_dataset)
            elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
                # Use a loop for TPUs when drop_last is False to have all batches have the same size.
                return DistributedSamplerWithLoop(
                    self.train_dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                )
            else:
                return DistributedSampler(
                    self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
                )
            
def main(): pass


if __name__ == "__main__":
    main()
