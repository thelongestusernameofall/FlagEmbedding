#!/bin/bash

unset http_proxy https_proxy
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

base_model=../bge-large-zh
out_model=../bge-large-zh-911
train_data=../all-0911-hn.jsonl

batch_size=160
epochs=10


cuda_visible_devices="$CUDA_VISIBLE_DEVICES"
devices=($(echo "$cuda_visible_devices" | tr ',' ' '))
num_devices="${#devices[@]}"

torchrun --nproc_per_node $num_devices \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir ${out_model} \
--model_name_or_path ${base_model} \
--train_data ${train_data} \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs ${epochs} \
--per_device_train_batch_size ${batch_size} \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 64 \
--passage_max_len 256 \
--train_group_size 2 \
--negatives_cross_device
