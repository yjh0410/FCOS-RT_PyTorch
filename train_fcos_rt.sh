python train.py \
        --cuda \
        -v fcos_rt \
        -bk r50 \
        --img_size 640 \
        --lr 0.01 \
        --batch_size 16 \
        --schedule 4 \
        --freeze_bn \
        --aug strong