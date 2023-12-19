#!/bin/bash
unset http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#base_model=../bge-large-zh-1029-v4
base_model=../bge-large-zh-1206-v2
action_dict=../all-1218-action-dict.json
output_file=../1219-hn-1.jsonl
common_query=../zhihu-sample-200000.jsonl
batch_size=4000

python -m FlagEmbedding.baai_general_embedding.finetune.hn_simon \
         --model_name_or_path ${base_model} \
         --input_file ${action_dict} \
         --output_file ${output_file} \
         --candidate_pool ${common_query} \
         --range_for_sampling 0-100 \
         --negative_number 10 \
         --batch_size ${batch_size} \
	 --score_threshold 0.55 \
         --use_gpu_for_searching
