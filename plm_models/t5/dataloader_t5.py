
import os

import pandas as pd
import json
from fastNLP import DataSet, Instance, logger
from fastNLP.io import Loader, DataBundle
from transformers import T5Tokenizer


PLM_MAX_LENGTH = 512


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=8)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids'],
        'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings


class BaseTCLoader(Loader):

    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens

    def convert_examples(self, example):
        raise NotImplementedError

    def pre_process(self, data_set: DataSet) -> DataSet:
        ds = DataSet()
        for example in data_set:
            example = self.convert_examples(example)
            input_encodings = self.tokenizer.encode_plus(example['input_text'])
            target_encodings = self.tokenizer.encode_plus(example['target_text'])
            ground_truth_encodings = self.tokenizer.encode_plus(example['ground_truth'])

            encodings = {
                'input_ids': input_encodings['input_ids'][: PLM_MAX_LENGTH],
                'attention_mask': input_encodings['attention_mask'][: PLM_MAX_LENGTH],
                'decoder_input_ids': target_encodings['input_ids'],
                'decoder_attention_mask': target_encodings['attention_mask'],
                'target_ids': ground_truth_encodings['input_ids'],
                'labels': ground_truth_encodings['input_ids'][1],
            }
            ins = Instance(**encodings)
            ds.append(ins)

        ds.set_input('input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'target_ids')
        ds.set_target('labels')
        return ds


class NLILoader(BaseTCLoader):

    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {}

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


class SST2Loader(BaseTCLoader):
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

