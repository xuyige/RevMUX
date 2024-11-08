

from copy import deepcopy
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertConfig,
    BertTokenizer,
)
from fastNLP import logger

from revmux.plm_models.bert.modeling_bert import BertLMHeadModel, BertForSequenceClassification


class SoftPrompt(nn.Module):

    def __init__(self, n_prompt_tokens=50, model_dim=512):
        super().__init__()
        self.n_prompt_tokens = n_prompt_tokens
        self.prompt = nn.Parameter(torch.randn(n_prompt_tokens, model_dim), requires_grad=True)

    def forward(self, input_embeds):
        batch_size = input_embeds.size(0)
        new_input_embeds = input_embeds + 0
        for idx in range(batch_size):
            new_input_embeds[idx, : self.n_prompt_tokens] = new_input_embeds[idx, : self.n_prompt_tokens] + self.prompt
        return new_input_embeds


class PromptTuningBaseline(nn.Module):

    def __init__(self, model_name, n_prompt_tokens=50):
        super().__init__()
        if model_name in ['bert-base-uncased', 'bert-large-uncased']:
            self.back_bone = 'bert'
            self.config = BertConfig.from_pretrained(model_name)
            self.config.d_model = self.config.hidden_size
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertLMHeadModel.from_pretrained(
                model_name,
                config=self.config,
            )

        self.prompt = SoftPrompt(n_prompt_tokens, self.config.d_model)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, target_ids=None):
        original_input_embeds = self.model.get_input_embeddings()(input_ids)
        input_embeds_with_prompt = self.prompt(original_input_embeds)

        result_dict = self.model(
            inputs_embeds=input_embeds_with_prompt,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
        logits = result_dict['logits'][:, 1]

        return_dict = {'logits': logits}
        if target_ids is not None:
            loss = self.loss_func(logits, target_ids[:, 1])
            return_dict['loss'] = loss

        return return_dict


class VanillaAdapterBatchInference(nn.Module):

    def __init__(self, model_name, target_index: List[int], n_prompt_tokens=50,
                 init_prompt=None, combine_first=0, compose_size=2, compute_similarity=True,
                 use_ada_prompt=False):
        super().__init__()
        self.compose_size = compose_size

        self.back_bone = 'bert'
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.d_model = self.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            decompose_size=self.compose_size
        )
        self.ans_pos = 0

        if init_prompt is None:
            self.prompt = SoftPrompt(n_prompt_tokens, self.config.d_model)
        else:
            self.prompt = deepcopy(init_prompt)
        self.use_ada_prompt = use_ada_prompt
        if self.use_ada_prompt:
            self.ada_prompt = SoftPrompt(n_prompt_tokens, self.config.d_model)
        else:
            self.ada_prompt = None

        self.mapping = nn.Linear(self.config.d_model, self.config.d_model)
        self.adapter = nn.Sequential(
            nn.Linear(self.config.d_model * self.compose_size, self.config.d_model * 3),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.config.d_model * 3, self.config.d_model),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.cos_sim_func = nn.CosineSimilarity(dim=-1)
        self.target_index = target_index

        self.combine_first = combine_first
        self.mode = 'normal'
        self.compute_cos_similarity = compute_similarity

    def compute_similarity(self, x1, x2, attention_mask1, attention_mask2):
        mapping_loss_f = self.cos_sim_func(x1, x2.detach())
        loss_mask_f = torch.ones_like(mapping_loss_f.detach())  # [seq_len]
        loss_mask_f.index_fill_(dim=0, index=(mapping_loss_f < 0).long(), value=-1.)
        mapping_loss_f = mapping_loss_f * loss_mask_f * attention_mask1.float()
        attention_mask1[0] = 1
        mapping_loss_f = mapping_loss_f.sum() / attention_mask1.sum().float()

        mapping_loss_b = self.cos_sim_func(x2, x1.detach())
        loss_mask_b = torch.ones_like(mapping_loss_b)  # [seq_len]
        loss_mask_b.index_fill_(dim=0, index=(mapping_loss_b < 0).long(), value=-1.)
        mapping_loss_b = mapping_loss_b * loss_mask_b * attention_mask2.float()
        attention_mask2[0] = 1
        mapping_loss_b = mapping_loss_b.sum() / attention_mask2.sum().float()

        return (mapping_loss_f + mapping_loss_b) / 2

    def compute_one_division(
            self, input_embeds_with_prompt, attention_mask, decoder_input_ids,
            decoder_attention_mask, target_ids=None
    ):
        batch_size, seq_len, d_model = input_embeds_with_prompt.size()

        if not self.training:
            if self.mode in ['teacher_only']:
                result_dict = self.model(
                    inputs_embeds=input_embeds_with_prompt,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decompose=False,
                    encoder_from_layer=0,
                    encoder_to_layer=-1,
                    decoder_from_layer=0,
                    decoder_to_layer=-1
                )
                logits = result_dict['logits']
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(1)
                logits_mask = torch.zeros_like(logits) - 1e5
                logits_mask.index_fill_(2, torch.tensor(self.target_index).to(logits.device), 0)
                logits = logits + logits_mask
                return {'logits': logits[:, self.ans_pos]}

        combine_result_dict = self.model(
            inputs_embeds=input_embeds_with_prompt,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decompose=False,
            encoder_from_layer=0,
            encoder_to_layer=self.combine_first,
            decoder_from_layer=0,
            decoder_to_layer=0
        )
        input_embeds_with_prompt = combine_result_dict['encoder_last_hidden_state']
        # [batch_size, seq_len, hidden_size]
        if self.use_ada_prompt:
            input_embeds_with_prompt = self.ada_prompt(input_embeds_with_prompt)

        new_input_embeds_with_prompt = self.mapping(input_embeds_with_prompt)
        new_attention_mask = attention_mask

        if batch_size % self.compose_size == 0:
            pad_size = 0
        else:
            pad_size = self.compose_size - batch_size % self.compose_size

        new_batch_size = batch_size // self.compose_size
        if pad_size > 0:
            new_input_embeds_with_prompt = torch.cat(
                [new_input_embeds_with_prompt,
                 torch.zeros_like(new_input_embeds_with_prompt)[: pad_size]],
                dim=0)
            new_attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask)[: pad_size]], dim=0)
            new_batch_size += 1

        new_input_embeds_with_prompt = new_input_embeds_with_prompt.view(
            new_batch_size, self.compose_size, seq_len, d_model)

        if self.training:
            original_new_input_embeds_with_prompt = new_input_embeds_with_prompt.clone()
            # [bacth // compose_size, compose_size, seq_len, d_model]
            original_new_attention_mask = new_attention_mask.contiguous()
            original_new_attention_mask = original_new_attention_mask.view(new_batch_size, self.compose_size, seq_len)
        else:
            original_new_attention_mask = 0

        new_input_embeds_with_prompt = new_input_embeds_with_prompt.permute(0, 2, 3, 1).contiguous().view(
            new_batch_size, seq_len, d_model * self.compose_size)
        new_input_embeds_with_prompt = self.adapter(new_input_embeds_with_prompt)

        new_attention_mask = new_attention_mask.view(
            new_batch_size, self.compose_size, seq_len)
        new_attention_mask = (new_attention_mask.sum(dim=1, keepdims=False) > 0).long()
        new_decoder_input_ids = decoder_input_ids[: new_batch_size]
        new_decoder_attention_mask = decoder_attention_mask[: new_batch_size]

        student_result_dict = self.model(
            inputs_embeds=new_input_embeds_with_prompt,
            attention_mask=new_attention_mask,
            decoder_input_ids=new_decoder_input_ids,
            decoder_attention_mask=new_decoder_attention_mask,
            decompose=True,
            encoder_from_layer=self.combine_first,
            encoder_to_layer=-1,
            decoder_from_layer=0,
            decoder_to_layer=-1,
        )

        original_student_sequence_output = student_result_dict['sequence_output'].permute(0, 2, 1, 3).contiguous()
        o_s_s_o_size = original_student_sequence_output.size()
        # [bacth // compose_size, compose_size, seq_len, d_model]
        original_student_logits = student_result_dict['logits']
        if len(original_student_logits.shape) == 3:
            original_student_logits = original_student_logits.unsqueeze(1)
        original_student_logits = original_student_logits.permute(0, 2, 1, 3).contiguous()
        o_s_l_size = original_student_logits.size()
        # [batch // compose_size, compose_size, seq_len, vocab_size]

        student_sequence_output = original_student_sequence_output.view(
            new_batch_size * self.compose_size, o_s_s_o_size[2], o_s_s_o_size[3])[: batch_size]
        # [batch, decoder_seq_len, d_model]
        student_logits = original_student_logits.view(
            new_batch_size * self.compose_size, o_s_l_size[2], o_s_l_size[3])[: batch_size]
        # [batch, decoder_seq_len, vocab_size]

        logits_mask = torch.zeros_like(student_logits[:, 0: 1]) - 1e5
        logits_mask.index_fill_(dim=2, index=torch.tensor(self.target_index).to(student_logits.device), value=0)
        student_logits[:, self.ans_pos: self.ans_pos + 1] += logits_mask

        info_nce_loss = 0.
        ce_loss = 0.
        final_mapping_loss = 0.
        return_dict = {
            'logits': student_logits[:, self.ans_pos],
            'loss': info_nce_loss + ce_loss + final_mapping_loss
        }

        if self.training:
            ce_loss = self.ce_loss_func(student_logits[:, self.ans_pos], target_ids[:, 1])

            with torch.no_grad():
                teacher_result_dict = self.model(
                    inputs_embeds=input_embeds_with_prompt,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decompose=False,
                    encoder_from_layer=self.combine_first,
                    encoder_to_layer=-1,
                    decoder_from_layer=0,
                    decoder_to_layer=-1,
                )
                teacher_sequence_output = teacher_result_dict['sequence_output']

            info_nce_label = torch.arange(batch_size).to(target_ids.device)

            normalize_student_sequence_output = F.normalize(student_sequence_output[:, self.ans_pos], dim=1)
            normalize_teacher_sequence_output = F.normalize(teacher_sequence_output[:, self.ans_pos], dim=1)

            inner_dot_student_teacher = torch.matmul(
                normalize_student_sequence_output, normalize_teacher_sequence_output.permute(1, 0)
            )
            info_nce_loss = \
                info_nce_loss + self.ce_loss_func(inner_dot_student_teacher * 20, info_nce_label)

            if self.compute_cos_similarity:
                new_batch_size, compose_size = original_new_input_embeds_with_prompt.size()[: 2]
                total_mapping_loss = 0.
                mapping_count = 0
                for idx in range(new_batch_size):
                    for i in range(compose_size):
                        for j in range(i + 1, compose_size):
                            total_mapping_loss = total_mapping_loss + self.compute_similarity(
                                original_new_input_embeds_with_prompt[idx, i],
                                original_new_input_embeds_with_prompt[idx, j],
                                original_new_attention_mask[idx, i],
                                original_new_attention_mask[idx, j]
                            )
                            mapping_count += 1
                final_mapping_loss = total_mapping_loss / max(mapping_count, 1)

            return_dict['loss'] = ce_loss + info_nce_loss * 0.5 + final_mapping_loss

        return return_dict

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, target_ids=None):
        original_input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.back_bone in ['bert']:
            input_embeds_with_prompt = original_input_embeds
        else:
            input_embeds_with_prompt = self.prompt(original_input_embeds)

        return_dict = self.compute_one_division(input_embeds_with_prompt, attention_mask,
                                                decoder_input_ids, decoder_attention_mask,
                                                target_ids)

        return return_dict


class ReversibleBatchInference(nn.Module):

    def __init__(self, model_name, target_index: List[int], n_prompt_tokens=50,
                 init_prompt=None, combine_first=0, compose_size=2, compute_similarity=1,
                 use_ada_prompt=False, invertible_decompose=False, lambda_info_nce=0.25):
        super().__init__()
        self.compose_size = compose_size

        self.back_bone = 'bert'
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.config.d_model = self.config.hidden_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            decompose_size=self.compose_size
        )
        self.ans_pos = 0
        self.lambda_info_mce = lambda_info_nce

        assert self.config.d_model % self.compose_size == 0, 'The model size is not divisible by the compose size.'
        self.dim_instance = self.config.d_model // self.compose_size

        if init_prompt is None:
            self.prompt = SoftPrompt(n_prompt_tokens, self.config.d_model)
        else:
            self.prompt = deepcopy(init_prompt)
        self.use_ada_prompt = use_ada_prompt
        if self.use_ada_prompt:
            self.ada_prompt = SoftPrompt(n_prompt_tokens, self.config.d_model)
        else:
            self.ada_prompt = None

        self.down_projection = nn.Linear(self.config.d_model, self.dim_instance)
        self.f_adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim_instance, self.config.d_model),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(self.config.d_model, self.dim_instance),
                nn.ReLU(),
                nn.Dropout(p=0.1),
            )
            for _ in range(self.compose_size)]
        )

        self.ce_loss_func = nn.CrossEntropyLoss()
        self.cos_sim_func = nn.CosineSimilarity(dim=-1)
        self.target_index = target_index

        self.combine_first = combine_first
        self.invertible_decompose = invertible_decompose
        self.mode = 'normal'
        self.compute_cos_similarity = compute_similarity

    def compute_similarity(self, x1, x2, attention_mask1, attention_mask2):
        mapping_loss_f = self.cos_sim_func(x1, x2.detach())
        loss_mask_f = torch.ones_like(mapping_loss_f.detach())  # [seq_len]
        loss_mask_f.index_fill_(dim=0, index=(mapping_loss_f < 0).long(), value=-1.)
        if self.compute_cos_similarity == 1:
            mapping_loss_f = mapping_loss_f * loss_mask_f * attention_mask1.float()
        elif self.compute_cos_similarity == -1:
            mapping_loss_f = 1. - mapping_loss_f * loss_mask_f
            mapping_loss_f = mapping_loss_f * attention_mask1.float()
        else:
            raise NotImplementedError
        attention_mask1[0] = 1
        mapping_loss_f = mapping_loss_f.sum() / attention_mask1.sum().float()

        mapping_loss_b = self.cos_sim_func(x2, x1.detach())
        loss_mask_b = torch.ones_like(mapping_loss_b)  # [seq_len]
        loss_mask_b.index_fill_(dim=0, index=(mapping_loss_b < 0).long(), value=-1.)
        if self.compute_cos_similarity == 1:
            mapping_loss_b = mapping_loss_b * loss_mask_b * attention_mask2.float()
        elif self.compute_cos_similarity == -1:
            mapping_loss_b = 1. - mapping_loss_b * loss_mask_b
            mapping_loss_b = mapping_loss_b * attention_mask2.float()
        else:
            raise NotImplementedError
        attention_mask2[0] = 1
        mapping_loss_b = mapping_loss_b.sum() / attention_mask2.sum().float()

        return (mapping_loss_f + mapping_loss_b) / 2

    def compute_invertible_representations(self, new_input_embeds_with_prompt):
        if self.compose_size == 1:
            return new_input_embeds_with_prompt

        x_list = [new_input_embeds_with_prompt[:, -2]]
        for idx in range(self.compose_size):
            x = new_input_embeds_with_prompt[:, self.compose_size - idx - 1] + self.f_adapter[idx](x_list[-1])
            x_list.append(x)

        new_input_embeds_with_prompt = torch.cat([x.unsqueeze(1) for x in x_list[1:]], dim=1)

        return new_input_embeds_with_prompt

    def compute_one_division(
            self, input_embeds_with_prompt, attention_mask, decoder_input_ids,
            decoder_attention_mask, target_ids=None
    ):
        batch_size, seq_len, d_model = input_embeds_with_prompt.size()

        if not self.training:
            if self.mode in ['teacher_only']:
                result_dict = self.model(
                    inputs_embeds=input_embeds_with_prompt,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decompose=False,
                    encoder_from_layer=0,
                    encoder_to_layer=-1,
                    decoder_from_layer=0,
                    decoder_to_layer=-1
                )
                logits = result_dict['logits']
                if len(logits.shape) == 2:
                    logits = logits.unsqueeze(1)
                logits_mask = torch.zeros_like(logits) - 1e5
                logits_mask.index_fill_(2, torch.tensor(self.target_index).to(logits.device), 0)
                logits = logits + logits_mask
                return {'logits': logits[:, self.ans_pos]}

        combine_result_dict = self.model(
            inputs_embeds=input_embeds_with_prompt,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decompose=False,
            encoder_from_layer=0,
            encoder_to_layer=self.combine_first,
            decoder_from_layer=0,
            decoder_to_layer=0
        )
        input_embeds_with_prompt = combine_result_dict['encoder_last_hidden_state']
        # [batch_size, seq_len, hidden_size]
        if self.use_ada_prompt:
            input_embeds_with_prompt = self.ada_prompt(input_embeds_with_prompt)

        new_input_embeds_with_prompt = self.down_projection(input_embeds_with_prompt)
        new_attention_mask = attention_mask

        if batch_size % self.compose_size == 0:
            pad_size = 0
        else:
            pad_size = self.compose_size - batch_size % self.compose_size

        new_batch_size = batch_size // self.compose_size
        if pad_size > 0:
            new_input_embeds_with_prompt = torch.cat(
                [new_input_embeds_with_prompt,
                 torch.zeros_like(new_input_embeds_with_prompt)[: pad_size]],
                dim=0)
            new_attention_mask = torch.cat([attention_mask, torch.zeros_like(attention_mask)[: pad_size]], dim=0)
            new_batch_size += 1
            # new_input_embeds_with_prompt: [batch, seq_len, d_model // compose_size]

        new_input_embeds_with_prompt = new_input_embeds_with_prompt.contiguous().view(
            new_batch_size, self.compose_size, seq_len, d_model // self.compose_size
        )
        # [new_batch_size, compose_size, seq_len, d_model // compose_size]

        if self.training:
            original_new_input_embeds_with_prompt = new_input_embeds_with_prompt.clone()
            # [bacth // compose_size, compose_size, seq_len, d_model // compose_size]
            original_new_attention_mask = new_attention_mask.contiguous()
            original_new_attention_mask = original_new_attention_mask.view(new_batch_size, self.compose_size, seq_len)
        else:
            original_new_attention_mask = 0

        new_input_embeds_with_prompt = self.compute_invertible_representations(new_input_embeds_with_prompt)

        new_attention_mask = new_attention_mask.view(
            new_batch_size, self.compose_size, seq_len)
        new_attention_mask = (new_attention_mask.sum(dim=1, keepdims=False) > 0).long()
        new_decoder_input_ids = decoder_input_ids[: new_batch_size]
        new_decoder_attention_mask = decoder_attention_mask[: new_batch_size]

        new_input_embeds_with_prompt = new_input_embeds_with_prompt.permute(0, 2, 3, 1).contiguous().view(
            new_batch_size, seq_len, d_model)

        student_result_dict = self.model(
            inputs_embeds=new_input_embeds_with_prompt,
            attention_mask=new_attention_mask,
            decoder_input_ids=new_decoder_input_ids,
            decoder_attention_mask=new_decoder_attention_mask,
            decompose=True,
            decompose_invertible_adapter=self.f_adapter if self.invertible_decompose else None,
            encoder_from_layer=self.combine_first,
            encoder_to_layer=-1,
            decoder_from_layer=0,
            decoder_to_layer=-1,
        )

        original_student_sequence_output = student_result_dict['sequence_output'].permute(0, 2, 1, 3).contiguous()
        o_s_s_o_size = original_student_sequence_output.size()
        # [bacth // compose_size, compose_size, seq_len, d_model]
        original_student_logits = student_result_dict['logits']
        if len(original_student_logits.shape) == 3:
            original_student_logits = original_student_logits.unsqueeze(1)
        original_student_logits = original_student_logits.permute(0, 2, 1, 3).contiguous()
        o_s_l_size = original_student_logits.size()
        # [batch // compose_size, compose_size, seq_len, vocab_size]

        student_sequence_output = original_student_sequence_output.view(
            new_batch_size * self.compose_size, o_s_s_o_size[2], o_s_s_o_size[3])[: batch_size]
        # [batch, decoder_seq_len, d_model]
        student_logits = original_student_logits.view(
            new_batch_size * self.compose_size, o_s_l_size[2], o_s_l_size[3])[: batch_size]
        # [batch, decoder_seq_len, vocab_size]

        logits_mask = torch.zeros_like(student_logits[:, 0: 1]) - 1e5
        logits_mask.index_fill_(dim=2, index=torch.tensor(self.target_index).to(student_logits.device), value=0)
        student_logits[:, self.ans_pos: self.ans_pos + 1] += logits_mask

        info_nce_loss = 0.
        ce_loss = 0.
        final_mapping_loss = 0.
        return_dict = {
            'logits': student_logits[:, self.ans_pos],
            'loss': info_nce_loss + ce_loss + final_mapping_loss
        }

        if self.training:
            ce_loss = self.ce_loss_func(student_logits[:, self.ans_pos], target_ids[:, 1])

            with torch.no_grad():
                teacher_result_dict = self.model(
                    inputs_embeds=input_embeds_with_prompt,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decompose=False,
                    encoder_from_layer=self.combine_first,
                    encoder_to_layer=-1,
                    decoder_from_layer=0,
                    decoder_to_layer=-1,
                )
                teacher_sequence_output = teacher_result_dict['sequence_output']

            info_nce_label = torch.arange(batch_size).to(target_ids.device)

            normalize_student_sequence_output = F.normalize(student_sequence_output[:, self.ans_pos], dim=1)
            normalize_teacher_sequence_output = F.normalize(teacher_sequence_output[:, self.ans_pos], dim=1)

            inner_dot_student_teacher = torch.matmul(
                normalize_student_sequence_output, normalize_teacher_sequence_output.permute(1, 0)
            )
            info_nce_loss = \
                info_nce_loss + self.ce_loss_func(inner_dot_student_teacher * 20, info_nce_label)

            if self.compute_cos_similarity != 0:
                new_batch_size, compose_size = original_new_input_embeds_with_prompt.size()[: 2]
                total_mapping_loss = 0.
                mapping_count = 0
                for idx in range(new_batch_size):
                    for i in range(compose_size):
                        for j in range(i + 1, compose_size):
                            total_mapping_loss = total_mapping_loss + self.compute_similarity(
                                original_new_input_embeds_with_prompt[idx, i],
                                original_new_input_embeds_with_prompt[idx, j],
                                original_new_attention_mask[idx, i],
                                original_new_attention_mask[idx, j]
                            )
                            mapping_count += 1
                final_mapping_loss = total_mapping_loss / max(mapping_count, 1)

            return_dict['loss'] = ce_loss + info_nce_loss * self.lambda_info_mce + final_mapping_loss

        return return_dict

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, target_ids=None):
        original_input_embeds = self.model.get_input_embeddings()(input_ids)
        if self.back_bone in ['bert']:
            input_embeds_with_prompt = original_input_embeds
        else:
            input_embeds_with_prompt = self.prompt(original_input_embeds)

        return_dict = self.compute_one_division(input_embeds_with_prompt, attention_mask,
                                                decoder_input_ids, decoder_attention_mask,
                                                target_ids)

        return return_dict
