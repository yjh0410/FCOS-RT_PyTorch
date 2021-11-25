python train.py \
        --cuda \
        -d coco \
        -v fcos_rt \
        -bk r50 \
        --img_size 512 \
        --lr 0.01 \
        --batch_size 16 \
        --schedule 4 \
        --multi_scale
