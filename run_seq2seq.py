#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
import torch
from extraction.event_schema import EventSchema
from extraction.extraction_metrics import decoding_format_dict
from extraction.calc_header_f_score_UFET import get_extract_metrics
from seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments, ConstraintSeq2SeqTrainer

# if not os.path.exists("~/nltk_data/tokenizers/punkt"):
#     print('Start download punk')
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
labels_full_names = {
    "naphtha": "Naphtha",
    "lead": "Lead",
    "zinc": "Zinc",
    "gas": "Natural Gas",
    "potato": "Potato",
    "alum": "Aluminum",
    "corn": "Corn",
    "heat": "Heating Oil",
    "groundnut": "Groundnut",
    "cotton": "Cotton",
    "earn": "Earnings",
    "coffee": "Coffee",
    "carcass": "Carcass",
    "soy-oil": "Soybean Oil",
    "bop": "Balance of Payments",
    "wheat": "Wheat",
    "reserves": "Reserves",
    "rape-oil": "Rapeseed Oil",
    "soy-meal": "Soybean Meal",
    "lumber": "Lumber",
    "wpi": "Wholesale Price Index",
    "gold": "Gold",
    "nickel": "Nickel",
    "lin-oil": "Linseed Oil",
    "l-cattle": "Live Cattle",
    "sunseed": "Sunflower Seed",
    "rand": "South African Rand",
    "lei": "Lei",
    "rye": "Rye",
    "fuel": "Fuel",
    "income": "Income",
    "iron-steel": "Iron and Steel",
    "palmkernel": "Palm Kernel",
    "hog": "Hog",
    "jet": "Jet Fuel",
    "soybean": "Soybean",
    "barley": "Barley",
    "sorghum": "Sorghum",
    "jobs": "Jobs",
    "sugar": "Sugar",
    "money-supply": "Money Supply",
    "rapeseed": "Rapeseed",
    "palm-oil": "Palm Oil",
    "rice": "Rice",
    "palladium": "Palladium",
    "money-fx": "Money Foreign Exchange",
    "nzdlr": "New Zealand Dollar",
    "livestock": "Livestock",
    "veg-oil": "Vegetable Oil",
    "gnp": "Gross National Product",
    "orange": "Orange",
    "cpi": "Consumer Price Index",
    "cotton-oil": "Cottonseed Oil",
    "yen": "Japanese Yen",
    "propane": "Propane",
    "acq": "Acquisitions",
    "castor-oil": "Castor Oil",
    "coconut": "Coconut",
    "oilseed": "Oilseed",
    "nkr": "Norwegian Krone",
    "meal-feed": "Meal Feed",
    "tea": "Tea",
    "sun-meal": "Sunflower Meal",
    "ipi": "Industrial Production Index",
    "pet-chem": "Petrochemicals",
    "instal-debt": "Installment Debt",
    "dfl": "Dutch Florin",
    "trade": "Trade",
    "grain": "Grain",
    "rubber": "Rubber",
    "sun-oil": "Sunflower Oil",
    "cpu": "Capacity Utilization",
    "dmk": "Deutsche Mark",
    "retail": "Retail",
    "cocoa": "Cocoa",
    "coconut-oil": "Coconut Oil",
    "ship": "Shipping",
    "nat-gas": "Natural Gas",
    "platinum": "Platinum",
    "tin": "Tin",
    "interest": "Interest Rates",
    "copper": "Copper",
    "strategic-metal": "Strategic Metals",
    "silver": "Silver",
    "copra-cake": "Copra Cake",
    "oat": "Oat",
    "housing": "Housing",
    "groundnut-oil": "Groundnut Oil",
    "dlr": "Dollar",
    "crude": "Crude Oil"
}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    # !!! must use non-fast version
    # fast: "<extra_id_0> <extra_id_1>" -> [32099, 3, 32098, 1]
    # non-fast: "<extra_id_0> <extra_id_1>" -> [32099, 32098, 1]
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    eos_penalty: float = field(
        default=1,
        metadata={
            "help": ""
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
                    "pegasus) or translation (or translation_{xx}_to_{yy})."
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge/sacreblue) on "
                    "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    source_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."})
    target_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."})
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("summarization") and not self.task.startswith(
                "translation") and not self.task.startswith('classification') and not self.task.startswith('emotion') and not self.task.startswith('entity_typing'):
            raise ValueError(
                "`task` should be summarization, summarization_{dataset}, translation or translation_{xx}_to_{yy}."
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

    # Start Code for Event Extraction
    decoding_format: str = field(
        default='tree',
        metadata={"help": "Decoding Format, valid in %s" %
                  decoding_format_dict.keys()}
    )
    event_schema: str = field(
        default=None, metadata={"help": "The input event schema file."}
    )
    # End Code for Event Extraction


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

event_extraction_name_mapping = {
    "classification": ("text", "table"),
    "emotion": ("text", "label"),
    "entity_typing": ("text", "label")
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, ConstraintSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # abla
    training_args.fp16 = training_args.is_fp16
    training_args.gradient_accumulation_steps = training_args.accumulate_steps
    print(model_args)
    print(data_args)
    print(training_args)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(
        training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `summary_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name,
                                data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # !!! Sometimes default max_length is setting to 20.
    config.max_length = data_args.max_target_length

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]
    from my_bart import BartForConditionalGeneration
    
    model = BartForConditionalGeneration.from_pretrained(
    # model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model.eos_penalty = model_args.eos_penalty
    if tokenizer.encode("<extra_id_0> <extra_id_1>") != [32099, 32098, 1]:
        # For non-t5 tokenizer
        lis = []
        for i in labels_full_names.values():
            lis += [i.lower()]
        tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<extra_id_0>", "<extra_id_1>", "<TRUE>", "<FALSE>", "<SEP>", "<start>", "<end>", "<title>", "<sentence>", "<context>"]})
        model.resize_token_embeddings(len(tokenizer))



        # 将自定义字符串转换为token ids
        embedding_list = []
        for i in lis:
            token_ids = tokenizer(i, return_tensors="pt", add_special_tokens=False).input_ids
            # 获取token embeddings
            with torch.no_grad():
                embeddings = model.get_input_embeddings()(token_ids)
                average_embedding = embeddings.mean(dim=1)
                embedding_list.append(average_embedding.squeeze())
        
        # 计算平均embedding
        tokenizer.add_special_tokens(
        {"additional_special_tokens": lis})
        model.resize_token_embeddings(len(tokenizer))
            # 获取特殊标记的索引，并将平均embedding赋值给该特殊标记
        for i, j in zip(lis, embedding_list): 
            special_token_id = tokenizer.convert_tokens_to_ids([i])
            with torch.no_grad():
                model.get_input_embeddings().weight[special_token_id] = j

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if data_args.task.startswith("translation"):
        if data_args.source_lang is not None:
            tokenizer.src_lang = data_args.source_lang
        if data_args.target_lang is not None:
            tokenizer.tgt_lang = data_args.target_lang

    # Start Code for Event Extraction
    if data_args.task.startswith("event"):
        decoding_type_schema = EventSchema.read_from_file(
            data_args.event_schema)
    else:
        decoding_type_schema = None
    # End Code for Event Extraction

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    source_lang, target_lang, text_column, summary_column = None, None, None, None

    if data_args.task.startswith("summarization"):
        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(
            data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column
    # Start Code for Event Extraction
    elif data_args.task.startswith("table") or data_args.task.startswith("emotion") or data_args.task.startswith("entity_typing") or data_args.task.startswith("classification"):
        dataset_columns = event_extraction_name_mapping.get(
            data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column
    # End Code for Event Extraction
    else:
        # Get the language codes for input/target.
        lang_search = re.match(
            "translation_([a-z]+)_to_([a-z]+)", data_args.task)
        if data_args.source_lang is not None:
            source_lang = data_args.source_lang.split("_")[0]
        else:
            assert (
                lang_search is not None
            ), "Provide a source language via --source_lang or rename your task 'translation_xx_to_yy'."
            source_lang = lang_search.groups()[0]

        if data_args.target_lang is not None:
            target_lang = data_args.target_lang.split("_")[0]
        else:
            assert (
                lang_search is not None
            ), "Provide a target language via --target_lang or rename your task 'translation_xx_to_yy'."
            target_lang = lang_search.groups()[1]

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        if data_args.task.startswith("translation"):
            inputs = [ex[source_lang] for ex in examples["translation"]]
            targets = [ex[target_lang] for ex in examples["translation"]]
        else:
            inputs = examples[text_column]
            targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(
                range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(
                range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(
            preds, skip_special_tokens=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=False)

        def clean_str(x_str):
            for to_remove_token in to_remove_token_list:
                x_str = x_str.replace(to_remove_token, '')
            return x_str.strip()

        decoded_preds = [clean_str(x) for x in decoded_preds]
        decoded_labels = [clean_str(x) for x in decoded_labels]

        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=decoding_type_schema,
            decoding_format=data_args.decoding_format,
        )

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = ConstraintSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        decoding_type_schema=decoding_type_schema,
        decoding_format=data_args.decoding_format,
        source_prefix=prefix,
    )

    # Training
    if training_args.do_train:
        # if last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        # elif os.path.isdir(model_args.model_name_or_path):
        #     checkpoint = model_args.model_name_or_path
        # else:
        #     checkpoint = None
        # TODO fix better about max_length
        checkpoint = None

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        # decoding_type_schema.write_to_file(
        #     os.path.join(
        #         training_args.output_dir,
        #         "event.schema",
        #     )
        # )

        output_train_file = os.path.join(
            training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))


    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        results = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams)
        results = {k: round(v, 4) for k, v in results.items()}

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )

        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
                )
                eval_preds = [pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
                              for pred in eval_preds]
                output_test_preds_file = os.path.join(
                    training_args.output_dir, "eval_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(eval_preds))

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

        output_test_result_file = os.path.join(
            training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=True
                )
                test_preds = [pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '').strip()
                              for pred in test_preds]
                output_test_preds_file = os.path.join(
                    training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
