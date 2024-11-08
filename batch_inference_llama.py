
import os
import argparse
import random
from copy import deepcopy

import torch
import torch.nn as nn
from fastNLP import (logger, Trainer, Tester, AccuracyMetric, RandomSampler,
                     SequentialSampler, EvaluateCallback, DataSet)
import fastNLP
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    Adafactor,
    AdamW
)

from revmux.utils import (set_random_seed, set_file_handler, get_data,
                          DataLoader, construct_true_few_shot_data,
                          set_input_target)
from revmux.nn_modules_llama import ReversibleBatchInference, VanillaAdapterBatchInference

from revmux.evaluation import evaluation_test
from revmux.optim import DoubleOptimizer

MAX_LENGTH=256


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/path/to/your/llama/dir')
    parser.add_argument('--task_name', type=str, choices=[
        'sst-2', 'rte', 'qnli', 'mrpc'
    ])
    parser.add_argument('--model_type', type=str, choices=[
        'pt', 'adapter', 'ora', 'revmux'
    ], default='revmux')
    parser.add_argument('--add_cos_sim', type=int, choices=[-1, 0, 1], default=1)
    parser.add_argument('--combine_first', type=int, default=3)
    parser.add_argument('--compose_size', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--adapter_lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--k_shot', type=int, default=0)
    parser.add_argument('--testing_time', type=int, default=10)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='/path/to/your/data/dir')
    parser.add_argument('--save_dir', type=str, default='.')
    arg = parser.parse_args()

    logger.info(f'Args: {arg.__dict__}')

    torch.autograd.set_detect_anomaly(True)

    model_name = arg.model_name
    task_name = arg.task_name
    add_cos_sim = arg.add_cos_sim
    combine_first = arg.combine_first
    compose_size = arg.compose_size
    seed = arg.random_seed
    adapter_lr = arg.adapter_lr
    batch_size = arg.batch_size
    k_shot = arg.k_shot
    testing_time = arg.testing_time
    n_epochs = arg.n_epochs
    model_type = arg.model_type
    data_dir = arg.data_dir
    save_dir = arg.save_dir

    task_name_to_data_dir = {
        'sst-2': 'SST-2',
        'rte': 'RTE',
        'qnli': 'QNLI',
        'mrpc': 'MRPC',
    }
    if task_name in task_name_to_data_dir:
        data_dir = os.path.join(data_dir, task_name_to_data_dir[task_name])

    back_bone_type = 'llama'

    label_convert = lambda x: x
    if back_bone_type in ['llama']:
        if task_name in ['sst-2']:
            target_index = [43324, 31587]  # llama3
        elif task_name in ['rte', 'qnli', 'mrpc']:
            target_index = [9891, 2201]  # [150, 4273]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    if back_bone_type in ['llama']:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError
    Loader = DataLoader[back_bone_type]

    data_bundle, data_loader = get_data(task_name=task_name, tokenizer=tokenizer,
                                        n_prompt_tokens=0, data_loader_dict=Loader,
                                        data_dir=data_dir)

    if task_name in ['rte', 'qnli', 'mrpc']:
        data_bundle.set_dataset(data_bundle.get_dataset('dev'), 'test')

    if task_name in ['sst-2', 'rte', 'qnli', 'mrpc']:
        train_data, dev_data, test_data = \
            data_bundle.get_dataset('train'), data_bundle.get_dataset('dev'), data_bundle.get_dataset('test')
    else:
        raise NotImplementedError

    if k_shot > 0:
        train_data, dev_data = construct_true_few_shot_data(train_data, k_shot)


    @fastNLP.cache_results(f'data_{task_name}_ft_llama.pkl')
    def load_large_scale_data():
        trn = data_loader.pre_process(train_data)
        val = data_loader.pre_process(dev_data)
        tst = data_loader.pre_process(test_data)
        return trn, val, tst

    if len(train_data) >= 70000:
        train_data, dev_data, test_data = load_large_scale_data()
    else:
        train_data = data_loader.pre_process(train_data)
        dev_data = data_loader.pre_process(dev_data)
        test_data = data_loader.pre_process(test_data)

    train_data.apply(lambda x: label_convert(x['labels']), new_field_name='labels',
                     is_input=True, is_target=True)
    train_data.apply(lambda x: [label_convert(k) for k in x['target_ids']],
                     new_field_name='target_ids', is_input=True, is_target=False)

    if task_name in ['qnli']:
        train_data.apply(lambda x: x['input_ids'][:MAX_LENGTH],
                         new_field_name='input_ids', is_input=True, is_target=False)
        train_data.apply(lambda x: x['attention_mask'][:MAX_LENGTH],
                         new_field_name='attention_mask', is_input=True, is_target=False)

    dev_data.apply(lambda x: label_convert(x['labels']), new_field_name='labels',
                   is_input=True, is_target=True)
    dev_data.apply(lambda x: [label_convert(k) for k in x['target_ids']],
                   new_field_name='target_ids', is_input=True, is_target=False)

    test_data.apply(lambda x: label_convert(x['labels']), new_field_name='labels',
                    is_input=True, is_target=True)
    test_data.apply(lambda x: [label_convert(k) for k in x['target_ids']],
                    new_field_name='target_ids', is_input=True, is_target=False)
    test_data = set_input_target(test_data, input_fields=['input_ids', 'attention_mask', 'target_ids', 'labels'])

    logger.info(f'start evaluating test set...')

    if task_name in ['sst-2']:
        dev_random_seed = 4141
    elif task_name in ['rte', 'qnli', 'mrpc']:
        dev_random_seed = 4040
    else:
        raise NotImplementedError
    set_random_seed(dev_random_seed)
    new_dev_data = fastNLP.DataSet()
    idx_list = [_ for _ in range(len(dev_data))]
    for _ in range(10):
        random.shuffle(idx_list)
        for idx in idx_list:
            new_dev_data.append(dev_data[idx])
    original_dev_data = deepcopy(dev_data)
    dev_data = set_input_target(new_dev_data, input_fields=['input_ids', 'attention_mask', 'target_ids', 'labels'])
    logger.info(f'successfully append dev data 10 times.')

    logger.info('Number of train data: {}'.format(len(train_data)))
    logger.info(f'Example:\n{train_data[0]}')
    logger.info('Number of dev data: {}'.format(len(dev_data)))
    logger.info(f'Example:\n{dev_data[0]}')
    logger.info('Number of test data: {}'.format(len(test_data)))
    logger.info(f'Example:\n{test_data[0]}')

    if task_name in ['sst-2']:
        set_random_seed(4141)
        idx_list = [_ for _ in range(len(test_data))]
        sst_test_data = DataSet()
        for _ in range(testing_time):
            random.shuffle(idx_list)
            for idx in idx_list:
                sst_test_data.append(test_data[idx])
        sst_test_data = set_input_target(sst_test_data, ['input_ids', 'attention_mask', 'target_ids', 'labels'])
        logger.info(f'successfully append test data 10 times for {task_name}.')
    else:
        sst_test_data = None

    set_random_seed(seed)
    callback_list = []

    torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_dtype(torch.half)

    if model_type in ['ora']:
        model = ReversibleBatchInference(
            model_name=model_name,
            combine_first=combine_first,
            target_index=target_index,
            compose_size=compose_size,
            compute_similarity=add_cos_sim,
        )
    elif model_type in ['revmux']:
        model = ReversibleBatchInference(
            model_name=model_name,
            target_index=target_index,
            combine_first=combine_first,
            compose_size=compose_size,
            compute_similarity=add_cos_sim,
            invertible_decompose=True,
        )
    elif model_type in ['adapter']:
        model = VanillaAdapterBatchInference(
            model_name=model_name,
            init_prompt=None,
            combine_first=combine_first,
            target_index=target_index,
            compose_size=compose_size,
            compute_similarity=(add_cos_sim == 1),
            use_ada_prompt=model_type in ['tp_vmbi'],
        )
    else:
        raise NotImplementedError


    # Function to initialize the weights and biases
    def init_linear_weights(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


    if model_type in ['ora', 'revmux']:
        model.down_projection.apply(init_linear_weights)
        model.f_adapter.apply(init_linear_weights)
        model.model.decompose.apply(init_linear_weights)
        model.model.decompose_up_projection.apply(init_linear_weights)
    elif model_type in ['adapter']:
        model.model.decompose.apply(init_linear_weights)
        model.adapter.apply(init_linear_weights)
        model.mapping.apply(init_linear_weights)
    else:
        raise NotImplementedError

    if task_name not in ['sst-2', 'rte', 'qnli']:
        logger.info(f'Check Teacher-Only Mode for {back_bone_type.upper()} backbone.')
        model.mode = 'teacher_only'
        tester = Tester(
            test_data,
            model,
            AccuracyMetric(pred='logits', target='labels'),
            batch_size=16,
            device=[_ for _ in range(torch.cuda.device_count())],
            # sampler=SequentialSampler()
            sampler=RandomSampler()
        )
        eval_results = tester.test()
        teacher_acc = eval_results['AccuracyMetric']['acc']
        logger.info(f'Accuracy of Teacher-Only mode is: {round(teacher_acc, 4)}')

    logger.info(f'successfully init `{model.__class__.__name__}` model.')

    no_decay = ['bias', 'LayerNorm.weight']
    if model_type in ['adapter']:
        adapter_parameters = [
            {'params': [p for n, p in model.model.decompose.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.model.decompose.named_parameters()
                        if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},

            {'params': [p for n, p in model.adapter.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.adapter.named_parameters() if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},

            {'params': [p for n, p in model.mapping.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.mapping.named_parameters() if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},
        ]
        adapter_optimizer = AdamW(
            adapter_parameters,
            lr=adapter_lr,
        )
        optimizer = adapter_optimizer
    elif model_type in ['ora', 'revmux']:
        adapter_parameters = [
            {'params': [p for n, p in model.model.decompose_up_projection.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.model.decompose_up_projection.named_parameters()
                        if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},

            {'params': [p for n, p in model.f_adapter.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.f_adapter.named_parameters() if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},

            {'params': [p for n, p in model.down_projection.named_parameters() if any([pn in n for pn in no_decay])],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.down_projection.named_parameters() if not any([pn in n for pn in no_decay])],
             'weight_decay': 0.01},
        ]
        if back_bone_type in ['llama']:
            new_adapter_parameters = [
                {'params': [p for n, p in model.model.decompose.named_parameters() if
                            any([pn in n for pn in no_decay])],
                 'weight_decay': 0.0},
                {'params': [p for n, p in model.model.decompose.named_parameters()
                            if not any([pn in n for pn in no_decay])],
                 'weight_decay': 0.01},
            ]
            adapter_parameters = adapter_parameters + new_adapter_parameters
        optimizer = AdamW(
            adapter_parameters,
            lr=adapter_lr,
        )
    else:
        raise NotImplementedError

    model.mode = 'normal'

    optimizer = DoubleOptimizer([optimizer])
    logger.info(f'Total Param: {optimizer.total_optim_param}')

    if task_name in ['sst-2']:
        callback_list.append(EvaluateCallback(sst_test_data))

    set_random_seed(seed)

    if task_name in ['rte']:
        validate_every = 24900
    elif task_name in ['sst-2']:
        validate_every = 33675
    elif task_name in ['qnli']:
        validate_every = 52350
    elif task_name in ['mrpc']:
        validate_every = 30000
    else:
        validate_every = None

    trainer = Trainer(
        train_data=train_data,
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        update_every=8,
        print_every=1,
        n_epochs=n_epochs,
        dev_data=dev_data,
        metrics=AccuracyMetric(pred='logits', target='labels'),
        metric_key='acc',
        validate_every=validate_every,
        device=[_ for _ in range(torch.cuda.device_count())],
        callbacks=callback_list,
        check_code_level=-1,
    )

    trainer.train()

    if task_name in ['sst-2']:
        evaluation_test(10, original_dev_data, model, task_name, set_batch_size=4,
                        input_fields=['input_ids', 'attention_mask', 'target_ids'])
    else:
        evaluation_test(testing_time, test_data, model, task_name, set_batch_size=4,
                        input_fields=['input_ids', 'attention_mask', 'target_ids'])

    if save_dir is not None:
        model.to('cpu')
        model_dict = {}

        if model_type in ['adapter']:
            model_dict['mapping'] = model.mapping.state_dict()
            model_dict['adapter'] = model.adapter.state_dict()
            model_dict['decompose'] = model.model.decompose.state_dict()

        if model_type in ['ora', 'revmux']:
            model_dict['down_projection'] = model.down_projection.state_dict()
            model_dict['f_adapter'] = model.f_adapter.state_dict()
            model_dict['decompose'] = model.model.decompose.state_dict()
            model_dict['decompose_up_projection'] = model.model.decompose_up_projection.state_dict()

        with open(f'{save_dir}/{model_name}-{model_type}-model.pkl', 'wb') as f:
            torch.save(model_dict, f)

        logger.info(f'successfully save model to file `{save_dir}/{model_name}-{model_type}-model.pkl`.')


