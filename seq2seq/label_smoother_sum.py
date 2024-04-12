#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from dataclasses import dataclass
import torch.nn.functional as F
import torch.nn as nn
@dataclass
class SumLabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (:obj:`float`, `optional`, defaults to 0.1):
            The label smoothing factor.
        ignore_index (:obj:`int`, `optional`, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """
    loss_type: str = 'set'
    epsilon: float = 0.1
    ignore_index: int = -100
    sep: int = 6
    tbc: int = 734
    end: int = 2
    
    @staticmethod
    def tensor_or(tensor1, tensor2):
        print(tensor1.shape)
        for i in range(len(tensor1)):
            for j in range(len(tensor1[0])):
                tensor1[i][j] = tensor1[i][j] or tensor2[i][j]
        return tensor1
            
    def __call__(self, labels, logits, branch_nums):
        if self.loss_type == 'set':
            log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
            ce_losses = torch.zeros(labels.shape[0]).to('cuda')
            seq_probs = torch.zeros(labels.shape[0]).to('cuda')
            branch_loss = torch.zeros(len(branch_nums)).to('cuda')

            for i in range(labels.shape[0]):
                ce_losses[i] = F.cross_entropy(target=labels[i:i+1], input=logits.transpose(1, 2)[i:i+1])
                seq_probs[i] = - sum([0 if  y == -100 else 1 for y in labels[i]]) * ce_losses[i]
            
            out_seq_probs = []
            start = 0
            tmp = torch.exp(seq_probs).detach().cpu().numpy()
            for i, branch_num in enumerate(branch_nums):
                out_seq_probs.append(tmp[start: start + branch_num])
                start += branch_num

            # for i in range(len(seq_probs)):
            #     if seq_probs[i] < -300:
            #         seq_probs[i] += (-300 - min(seq_probs[i]).detach())

            start = 0
            for i, branch_num in enumerate(branch_nums):
                branch_loss[i] = - torch.log(torch.mean(torch.exp(-ce_losses[start: start + branch_num])))
                start += branch_num

            loss = torch.mean(branch_loss)

            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)

            padding_mask = self.tensor_or(labels.eq(self.ignore_index), labels.eq(self.sep))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.tbc))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.end))
            smoothed_loss = log_probs.sum(dim=-1, keepdim=True)
            smoothed_loss.masked_fill_(padding_mask, 0.0)

            smoothed_loss = smoothed_loss.mean()  # / (num_active_elements * log_probs.shape[-1])
            eps_i = self.epsilon / log_probs.size(-1)
            return (1 - self.epsilon) * loss + eps_i * smoothed_loss, out_seq_probs
        
        if self.loss_type == 'vinyal':
            log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
            ce_losses = torch.zeros(labels.shape[0]).to('cuda')
            seq_probs = torch.zeros(labels.shape[0]).to('cuda')
            branch_loss = torch.zeros(len(branch_nums)).to('cuda')

            for i in range(labels.shape[0]):
                ce_losses[i] = F.cross_entropy(target=labels[i:i+1], input=logits.transpose(1, 2)[i:i+1])
                seq_probs[i] = - sum([0 if  y == -100 else 1 for y in labels[i]]) * ce_losses[i]
            
            out_seq_probs = []
            start = 0
            tmp = torch.exp(seq_probs).detach().cpu().numpy()
            for i, branch_num in enumerate(branch_nums):
                out_seq_probs.append(tmp[start: start + branch_num])
                start += branch_num

            # for i in range(len(seq_probs)):
            #     if seq_probs[i] < -300:
            #         seq_probs[i] += (-300 - min(seq_probs[i]).detach())

            start = 0
            for i, branch_num in enumerate(branch_nums):
                branch_loss[i] = torch.mean(ce_losses[start: start + branch_num])
                start += branch_num

            loss = torch.mean(branch_loss)

            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)

            padding_mask = self.tensor_or(labels.eq(self.ignore_index), labels.eq(self.sep))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.tbc))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.end))
            smoothed_loss = log_probs.sum(dim=-1, keepdim=True)
            smoothed_loss.masked_fill_(padding_mask, 0.0)

            smoothed_loss = smoothed_loss.mean()  # / (num_active_elements * log_probs.shape[-1])
            eps_i = self.epsilon / log_probs.size(-1)
            return (1 - self.epsilon) * loss + eps_i * smoothed_loss, out_seq_probs
        
        if self.loss_type == 'qin':
            log_probs = -torch.nn.functional.log_softmax(logits, dim=-1)
            ce_losses = torch.zeros(labels.shape[0]).to('cuda')
            seq_probs = torch.zeros(labels.shape[0]).to('cuda')
            branch_loss = torch.zeros(len(branch_nums)).to('cuda')

            for i in range(labels.shape[0]):
                ce_losses[i] = F.cross_entropy(target=labels[i:i+1], input=logits.transpose(1, 2)[i:i+1])
                seq_probs[i] = - sum([0 if  y == -100 else 1 for y in labels[i]]) * ce_losses[i]
            
            out_seq_probs = []
            start = 0
            tmp = torch.exp(seq_probs).detach().cpu().numpy()
            for i, branch_num in enumerate(branch_nums):
                out_seq_probs.append(tmp[start: start + branch_num])
                start += branch_num

            # for i in range(len(seq_probs)):
            #     if seq_probs[i] < -300:
            #         seq_probs[i] += (-300 - min(seq_probs[i]).detach())

            start = 0
            for i, branch_num in enumerate(branch_nums):
                branch_loss[i] = - torch.log(torch.sum(torch.exp(seq_probs[start: start + branch_num])))
                start += branch_num

            loss = torch.mean(branch_loss)

            if labels.dim() == log_probs.dim() - 1:
                labels = labels.unsqueeze(-1)

            padding_mask = self.tensor_or(labels.eq(self.ignore_index), labels.eq(self.sep))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.tbc))
            padding_mask = self.tensor_or(padding_mask, labels.eq(self.end))
            smoothed_loss = log_probs.sum(dim=-1, keepdim=True)
            smoothed_loss.masked_fill_(padding_mask, 0.0)

            smoothed_loss = smoothed_loss.mean()  # / (num_active_elements * log_probs.shape[-1])
            eps_i = self.epsilon / log_probs.size(-1)
            return (1 - self.epsilon) * loss + eps_i * smoothed_loss, out_seq_probs

