#!/bin/bash
unset http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

base_model=../bge-large-zh-911
action_dict=../all-1029-action-dict.json
output_file=../all-1029-hn.jsonl
common_query=../zhihu-sample-200000.jsonl
batch_size=4096

python -m FlagEmbedding.baai_general_embedding.finetune.hn_simon \
         --model_name_or_path ${base_model} \
         --input_file ${action_dict} \
         --output_file ${output_file} \
         --candidate_pool ${common_query} \
         --range_for_sampling 1-200 \
         --negative_number 30 \
         --batch_size ${batch_size} \
         --use_gpu_for_searching