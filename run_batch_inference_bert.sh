#!/usr/bin/env bash

export PYTHONPATH="${PYTHONPATH}:/path/to/your/workspace"

task_name="sst-2"
model_name="bert-base-uncased"
model_type="ibin"
batch_size=16
k_shot=0
n_epochs=10
testing_time=10
combine_first=16
compose_size=2
data_dir="/path/to/your/data/dir"
save_dir="."
adapter_lr="2e-5"


while [[ $# -gt 0 ]]; do
    case ${1} in
        --task_name)
            task_name="${2}"
            shift
            shift
            ;;
        --model_name)
            model_name="${2}"
            shift
            shift
            ;;
        --combine_first)
            combine_first=${2}
            shift
            shift
            ;;
        --model_type)
            model_type="${2}"
            shift
            shift
            ;;
        --k_shot)
            k_shot=${2}
            shift
            shift
            ;;
        --testing_time)
            testing_time="${2}"
            shift
            shift
            ;;
        --batch_size)
            batch_size=${2}
            shift
            shift
            ;;
        --adapter_lr)
            adapter_lr="${2}"
            shift
            shift
            ;;
        --n_epochs)
            n_epochs=${2}
            shift
            shift
            ;;
        --compose_size)
            compose_size=${2}
            shift
            shift
            ;;
        --data_dir)
            data_dir="${2}"
            shift
            shift
            ;;
        --save_dir)
            save_dir="${2}"
            shift
            shift
            ;;
        *)
            echo "cannot recognize ${1}"
            shift
            ;;
    esac
done

python batch_inference_bert.py \
  --task_name "${task_name}" \
  --model_name "${model_name}" \
  --model_type "${model_type}" \
  --k_shot ${k_shot} \
  --adapter_lr "${adapter_lr}" \
  --random_seed 42 \
  --batch_size ${batch_size} \
  --testing_time ${testing_time} \
  --n_epochs ${n_epochs} \
  --compose_size ${compose_size} \
  --combine_first ${combine_first} \
  --data_dir "${data_dir}" \
  --save_dir "${save_dir}"

