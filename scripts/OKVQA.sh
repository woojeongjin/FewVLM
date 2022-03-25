# The name of experiment
name=$2

output=snap/okvqa/$name

PYTHONPATH=$PYTHONPATH:./src \
CUDA_VISIBLE_DEVICES=$1 python src/okvqa.py \
        --train train \
        --valid train \
        --test val \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 200 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output  \
        --num_beams 5 \
        --batch_size 30 \
        --valid_batch_size 1000 \
        --load snap/pretrain/Epoch30 \
        ${@:3}

# bash scripts/OKVQA.sh 2 OKVQA --subsample --dataseed 42 --num_data 16 --test_only --prompt 3