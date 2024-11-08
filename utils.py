
import os
import random
import copy

import numpy as np
import torch
from fastNLP import DataSet, logger
import logging


from revmux.plm_models.t5.dataloader_t5 import (SST2Loader as T5_SST2, RTELoader,
                                                QNLILoader, MRPCLoader)
from revmux.plm_models.bert.dataloader_bert import (SST2Loader as Bert_SST2, RTELoader as Bert_RTE,
                                                    MRPCLoader as Bert_MRPC, QNLILoader as Bert_QNLI)
from revmux.plm_models.llama.dataloader_llama import (SST2Loader as LLAMA_SST2, RTELoader as LLAMA_RTE,
                                                      MRPCLoader as LLAMA_MRPC, QNLILoader as LLAMA_QNLI)
from revmux.fs_sampler import FewShotSampler

DataLoader = {
    't5': {'sst-2': T5_SST2, 'rte': RTELoader,
           'qnli': QNLILoader, 'mrpc': MRPCLoader,},
    'bert': {'sst-2': Bert_SST2, 'rte': Bert_RTE,
             'mrpc': Bert_MRPC, 'qnli': Bert_QNLI},
    'llama': {'sst-2': LLAMA_SST2, 'rte': LLAMA_RTE,
              'mrpc': LLAMA_MRPC, 'qnli': LLAMA_QNLI}
}


def set_input_target(
        data_set: DataSet,
        input_fields=['input_ids', 'attention_mask', 'decoder_input_ids', 'decoder_attention_mask', 'target_ids'],
        target_fields=['labels'],
) -> DataSet:
    for f in input_fields:
        data_set.set_input(f)
    for f in target_fields:
        data_set.set_target(f)
    return data_set


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Global seed set to {seed}")


def set_file_handler(path, level='INFO'):
    def _get_level(lv):
        if isinstance(lv, int):
            pass
        else:
            lv = lv.lower()
            lv = {'info': logging.INFO, 'debug': logging.DEBUG,
                  'warn': logging.WARN, 'warning': logging.WARN,
                  'error': logging.ERROR}[lv]
        return lv

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(path) == h.baseFilename:
                # file path already added
                return

    # File Handler
    if os.path.exists(path):
        assert os.path.isfile(path)
    dir_name = os.path.abspath(os.path.dirname(path))
    os.makedirs(dir_name, exist_ok=True)

    file_handler = logging.FileHandler(path, mode='a')
    file_handler.setLevel(_get_level(level))

    file_formatter = logging.Formatter('[%(asctime)s %(levelname)s]-'
                                       '[%(filename)s %(lineno)d] %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def get_data(task_name, tokenizer, n_prompt_tokens, data_loader_dict, data_dir='/media/yige/dataset/SST-2'):
    if task_name in ['sst-2', 'rte', 'snli', 'qnli', 'mrpc', 'qqp']:
        splits = ['train', 'dev', 'test']
    elif task_name in ['ag', 'imdb']:
        splits = ['train', 'test']
    else:
        raise NotImplementedError
    splits_dict = {'splits': splits}

    data_loader = data_loader_dict[task_name](tokenizer=tokenizer, n_prompt_tokens=n_prompt_tokens)
    data_bundle = data_loader.my_load(**splits_dict, data_dir=data_dir)
    return data_bundle, data_loader


def construct_true_few_shot_data(train_data, k_shot, dev_data=None):
    sampler = FewShotSampler(num_examples_per_label=k_shot if dev_data is not None else (k_shot * 2),
                             name_of_label_column='labels')
    new_train_data = sampler(train_data, seed=144)

    len_data = len(new_train_data)
    if dev_data is None:
        new_dev_data = new_train_data[-(len_data // 2):]
        new_train_data = new_train_data[:len_data // 2]
    else:
        new_dev_data = sampler(dev_data, seed=144)

    train_ds, dev_ds = DataSet(), DataSet()
    for e in new_train_data:
        train_ds.append(e)
    for e in new_dev_data:
        dev_ds.append(e)
    new_train_data = copy.deepcopy(train_ds)
    new_dev_data = copy.deepcopy(dev_ds)

    return new_train_data, new_dev_data

