# The name of experiment
name=$2

output=snap/nocaps/$name

PYTHONPATH=$PYTHONPATH:./src \
CUDA_VISIBLE_DEVICES=$1 python src/nocaps.py \
        --train karpathy_train \
        --valid karpathy_train \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 200 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output \
        --load snap/pretrain/Epoch30 \
        --num_beams 5 \
        --batch_size 30 \
        --valid_batch_size 100 \
        --caption_data dataset_coco \
        ${@:3}




# bash scripts/Nocaps.sh 2 nocaps --subsample --dataseed 42 --num_data 16 --prefix image --test_only 