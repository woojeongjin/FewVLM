# The name of experiment
name=$2

output=snap/vqa/$name

PYTHONPATH=$PYTHONPATH:./src \
CUDA_VISIBLE_DEVICES=$1 python src/vqa.py \
        --train train \
        --valid train \
        --test minival,nominival \
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



# bash scripts/VQA.sh 2 VQA --subsample --dataseed 42 --num_data 16 --test_only --prompt 3