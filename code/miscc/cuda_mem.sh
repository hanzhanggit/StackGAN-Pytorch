#!/usr/bin/bash
#precision=4 # FP32- 4bytes
#w=256
#h=256
#c=3
#batch=64
#embedding_dim=1024
#additional=100 # 100MB
#mb=$((($precision * (($w * $h * $c * $batch) + ($embedding_dim * $batch) + ($additional * 1048576))) / 1048576))
# The above setting took around 450MB space
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512, garbage_collection_threshold:0.8"
