
import os

import pandas as pd
import json
from fastNLP import DataSet, Instance, logger
from fastNLP.io import Loader, DataBundle
from transformers import AutoTokenizer

from v2.plm_models.llama.utils_llama import PLM_MAX_LENGTH


class BaseTCLoader(Loader):

    def __init__(self, tokenizer=None, model_id='/media/hdd/yige/llm/Llama3-8B-Instruction', instruct=True, **kwargs):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        else:
            self.tokenizer = tokenizer
        self.instruct = instruct
        self.prefix = 0
        self.postfix = 0
        self.task = None

    def convert_examples(self, example):
        raise NotImplementedError

    def pre_process(self, data_set: DataSet) -> DataSet:
        ds = DataSet()
        if self.task in ['rte']:
            plm_max_length = 32
        else:
            plm_max_length = PLM_MAX_LENGTH
        for example in data_set:
            example = self.convert_examples(example)
            if self.instruct:
                tokenized_dict = self.tokenizer.apply_chat_template(example['input_text'], return_dict=True)
                input_ids = \
                    tokenized_dict['input_ids'][: self.prefix] + \
                    tokenized_dict['input_ids'][self.prefix: -self.postfix][: plm_max_length] + \
                    tokenized_dict['input_ids'][-self.postfix:]
                attention_mask = \
                    tokenized_dict['attention_mask'][: self.prefix] + \
                    tokenized_dict['attention_mask'][self.prefix: -self.postfix][: plm_max_length] + \
                    tokenized_dict['attention_mask'][-self.postfix:]

                input_ids = input_ids[: -1]
                attention_mask = attention_mask[: -1]
            else:
                raise NotImplementedError

            ground_truth_encodings = self.tokenizer.encode_plus(example['ground_truth'])

            encodings = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'target_ids': ground_truth_encodings['input_ids'],
                'labels': ground_truth_encodings['input_ids'][1],
            }
            ins = Instance(**encodings)
            ds.append(ins)

        ds.set_input('input_ids', 'attention_mask', 'target_ids')
        ds.set_target('labels')
        return ds


class NLILoader(BaseTCLoader):

    def __init__(self, tokenizer=None, model_id='/media/hdd/yige/llm/Llama3-8B-Instruction', instruct=True, **kwargs):
        super().__init__(tokenizer, model_id, instruct)
        self.label2text = {}
        self.prefix = 37
        self.postfix = 0

    def convert_examples(self, example):
        if self.instruct:
            messages = [
                {'role': 'user', 'content': 'You are require to predict the two following sentences are entailment '
                                            'or not (yes or no). You should response yes or no, '
                                            'only one token is accepted.'},
                {'role': 'user', 'content': f'<|start of the sentence1|>: {example["sentence1"]} '
                                            f'<|end of the sentence1|>'},
                {'role': 'user', 'content': f'<|start of the sentence2|>: {example["sentence2"]} '
                                            f'<|end of the sentence2|>'},
                {'role': 'assistant', 'content': ''},
            ]
        else:
            raise NotImplementedError
        example['input_text'] = messages
        example['ground_truth'] = self.label2text[example['labels']]

        return example


class SST2Loader(BaseTCLoader):
    def __init__(self, tokenizer=None, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.label2text = {
            0: "negative",
            1: "positive",
        }
        self.prefix = 40
        # self.postfix = 45
        self.postfix = 0
        self.task = 'sst-2'

    def convert_examples(self, example):
        if self.instruct:
            messages = [
                {'role': 'user', 'content': 'You are require to predict the sentiment (positive or negative) '
                                            'to the following sentence. You should response positive or negative, '
                                            'only one token is accepted.'},
                {'role': 'user', 'content': f'<|start of the sentence|>: {example["sentence"]} '
                                            '<|end of the sentence|>'},
                # {'role': 'user', 'content': 'Repeat the task again: you are require to predict the sentiment '
                #                             '(positive or negative) to the aforementioned sentence. You should '
                #                             'response positive or negative, only one token is accepted.'},
                {'role': 'assistant', 'content': ''},
            ]
        else:
            raise NotImplementedError
        example['input_text'] = messages
        example['ground_truth'] = self.label2text[example['labels']]
        return example

    def _load(self, split, data_dir='/media/yige/dataset/SST-2') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.tsv')

        with open(file_name) as f:
            lines = f.readlines()[1:]

        ds = DataSet()
        for line in lines:
            text, label = line.split('\t')
            example = {'sentence': text, 'labels': int(label)}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/SST-2') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(NLILoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            'not_entailment': 'no',
            'entailment': 'yes',
        }
        self.task = 'rte'

    def _load(self, split, data_dir='/media/yige/dataset/RTE') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.tsv')

        with open(file_name) as f:
            lines = f.readlines()[1:]

        ds = DataSet()
        for line in lines:
            if split in ['test']:
                idx, text_a, text_b = line.split('\t')
                label = 'entailment'
            else:
                idx, text_a, text_b, label = line.split('\t')
            example = {'sentence1': text_a, 'sentence2': text_b, 'labels': label.strip()}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/RTE') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QNLILoader(NLILoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            'not_entailment': 'no',
            'entailment': 'yes',
        }
        self.task = 'qnli'

    def _load(self, split, data_dir='/media/yige/dataset/QNLI') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.tsv')

        with open(file_name) as f:
            lines = f.readlines()[1:]

        ds = DataSet()
        for line in lines:
            if split in ['test']:
                idx, text_a, text_b = line.split('\t')
                label = 'entailment'
            else:
                idx, text_a, text_b, label = line.split('\t')
            example = {'sentence1': text_a, 'sentence2': text_b, 'labels': label.strip()}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/QNLI') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(NLILoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            '0': 'no',
            '1': 'yes',
        }
        self.task = 'mrpc'

    def _load(self, split, data_dir='/media/yige/dataset/MRPC') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.tsv')

        with open(file_name) as f:
            lines = f.readlines()[1:]

        ds = DataSet()
        for line in lines:
            if split in ['test']:
                idx, id_1, id_2, text_a, text_b = line.split('\t')
                label = '1'
            else:
                label, id_1, id_2, text_a, text_b = line.split('\t')
            example = {'sentence1': text_a, 'sentence2': text_b, 'labels': label.strip()}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/MRPC') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class QQPLoader(NLILoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            '0': 'no',
            '1': 'yes',
        }

    def _load(self, split, data_dir='/media/yige/dataset/QQP') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.tsv')

        with open(file_name) as f:
            lines = f.readlines()[1:]

        ds = DataSet()
        for line in lines:
            if split in ['test']:
                idx, text_a, text_b = line.split('\t')
                label = '1'
            else:
                idx, id_1, id_2, text_a, text_b, label = line.split('\t')
            example = {'sentence1': text_a, 'sentence2': text_b, 'labels': label.strip()}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/QQP') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class SNLILoader(BaseTCLoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            'contradiction': 'no',
            'entailment': 'yes',
            'neutral': 'perhaps',
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = 'text: %s %s ? <extra_id_0> %s . </s>' % \
                                    (prompt, example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> <extra_id_0> </s>'
            example['labels'] = self.label2text[example['labels']]
            example['ground_truth'] = f'<pad> {example["labels"]} {example["labels"]} </s>'
        else:
            raise NotImplementedError

        return example

    def _load(self, split, data_dir='/media/yige/dataset/SNLI') -> DataSet:
        file_name = os.path.join(data_dir, f'snli_1.0_{split}.jsonl')

        ds = DataSet()
        with open(file_name) as f:
            for idx, line in enumerate(f):
                row = json.loads(line)

                text_a, text_b = row['sentence1'], row['sentence2']
                label = row['gold_label']
                if label.strip() in ['-']:
                    continue
                example = {'sentence1': text_a, 'sentence2': text_b, 'labels': label.strip()}
                ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/SNLI') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(BaseTCLoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            1: "world",
            2: "sports",
            3: "business",
            4: "tech"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s <extra_id_0> News: [Title] %s [Content] %s </s>' % \
                                    (prompt, example['sentence1'], example['sentence2'])
            example['target_text'] = '<pad> <extra_id_0> <extra_id_0> </s>'
            example['labels'] = self.label2text[example['labels']]
            example['ground_truth'] = f'<pad> {example["labels"]} {example["labels"]} </s>'
        else:
            raise NotImplementedError

        return example

    def _load(self, split, data_dir='/media/yige/dataset/AG') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.csv')

        data_frame = pd.read_csv(file_name, sep=',', header=None)

        ds = DataSet()
        for index, line in data_frame.iterrows():
            label, title, content = line
            example = {'sentence1': title, 'sentence2': content, 'labels': int(label)}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/AG') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class IMDBLoader(BaseTCLoader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__(tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
        self.label2text = {
            0: "negative",
            1: "positive",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = 'text: %s %s . </s>' % (prompt, example['sentence'])
            example['target_text'] = '<pad> <extra_id_0> <extra_id_0> </s>'
            example['labels'] = self.label2text[example['labels']]
            example['ground_truth'] = f'<pad> {example["labels"]} {example["labels"]} </s>'
        else:
            raise NotImplementedError

        return example

    def _load(self, split, data_dir='/media/yige/dataset/IMDB') -> DataSet:
        file_name = os.path.join(data_dir, f'{split}.csv')

        with open(file_name) as f:
            lines = f.readlines()

        ds = DataSet()
        for line in lines:
            label, text = line.split('\t')
            example = {'sentence': text.strip(), 'labels': int(label.strip())}
            ds.append(Instance(**example))

        logger.info(f'successfully load {split} set')
        return ds

    def my_load(self, splits, data_dir='/media/yige/dataset/IMDB') -> DataBundle:
        datasets = {name: self._load(name, data_dir) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

