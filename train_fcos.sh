python train.py \
        --cuda \
        -v fcos \
        -bk r50 \
        --img_size 640 \
        --lr 0.01 \
        --batch_size 16 \
        --schedule 1 \
        --freeze_bn \
        --no_warmup