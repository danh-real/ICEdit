
CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000104977.jpg \
    --instruction "remove the broccoli" \
    --seed 42 \

CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe_step2.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000104977.jpg \
    --step1-image COCO_train2014_000000104977.jpg_remove.jpg \
    --instruction "add a larger version of the broccoli from the left image" \
    --seed 42 \