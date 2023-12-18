#!/bin/bash

unset http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

base_model=../bge-large-zh-911
base_model=../bge-large-zh-1029-v4 # using before 1206
base_model=../bge-large-zh-1206-v2

out_model=../bge-large-zh-1218-v1
train_data=../1218-hn-1.jsonl

batch_size=60
epochs=20

master_port=29600

cuda_visible_devices="$CUDA_VISIBLE_DEVICES"
devices=($(echo "$cuda_visible_devices" | tr ',' ' '))
num_devices="${#devices[@]}"

torchrun --nproc_per_node $num_devices --master-port ${master_port} \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ${out_model} \
--model_name_or_path ${base_model} \
--train_data ${train_data} \
--learning_rate 8e-5 \
--fp16 \
--num_train_epochs ${epochs} \
--per_device_train_batch_size ${batch_size} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--save_steps 10000 \
--logging_steps 10 \
--warmup_ratio 0.003 \
--save_total_limit 3 \
--negatives_cross_device