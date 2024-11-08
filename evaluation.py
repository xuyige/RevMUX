
import random
import torch

from fastNLP import logger, DataSet, Tester, AccuracyMetric, SequentialSampler

from revmux.utils import set_random_seed, set_input_target


def evaluation_test(testing_time, test_data, model, task_name, set_batch_size=None,
                    input_fields=['input_ids', 'attention_mask', 'decoder_input_ids',
                                  'decoder_attention_mask', 'target_ids']):
    set_random_seed(4141)
    idx_list = [_ for _ in range(len(test_data))]
    test_data_list = []
    for _ in range(testing_time):
        random.shuffle(idx_list)
        test_data_list.append(DataSet())
        for idx in idx_list:
            test_data_list[-1].append(test_data[idx])
        test_data_list[-1] = set_input_target(test_data_list[-1], input_fields=input_fields)
    logger.info(f'successfully append test data {testing_time} times.')

    if set_batch_size is not None:
        batch_size = set_batch_size
    else:
        if task_name in ['rte']:
            batch_size = 16
        else:
            batch_size = 32
    model.mode = 'normal'
    model.eval()
    total_acc = 0.

    for t in range(testing_time):
        logger.info(f'{"=" * 30}')
        logger.info(f'Testing {t + 1}-th time ({len(test_data_list[t])} testing examples):')
        tester = Tester(
            test_data_list[t],
            model,
            AccuracyMetric(pred='logits', target='labels'),
            batch_size=batch_size,
            device=[_ for _ in range(torch.cuda.device_count())],
            sampler=SequentialSampler(),
            verbose=10,
        )
        logger.info(f'Tester.verbose: {tester.verbose}')
        eval_results = tester.test()
        acc = eval_results['AccuracyMetric']['acc']
        total_acc += acc

        logger.info(f'average acc: {round(total_acc / (t + 1), 6)}')

    model.mode = 'teacher_only'
    tester = Tester(
        test_data,
        model,
        AccuracyMetric(pred='logits', target='labels'),
        batch_size=batch_size,
        device=[_ for _ in range(torch.cuda.device_count())],
        sampler=SequentialSampler(),
        verbose=10
    )
    eval_results = tester.test()
    teacher_acc = eval_results['AccuracyMetric']['acc']
    logger.info(f'Accuracy of Teacher-Only mode is: {round(teacher_acc, 4)}')
