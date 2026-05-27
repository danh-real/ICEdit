CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
    --instruction "remove the bowl of fruit, then add a smaller version of the bowl of fruit at the same location" \
    --seed 42 \

# CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe_step1.py \
#     --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
#     --instruction "remove the bowl of fruit" \
#     --seed 42 \

# CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe_step2.py \
#     --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
#     --step1-image COCO_train2014_000000320451.jpg_remove.jpg \
#     --instruction "add one smaller version of the bowl of fruit from the left" \
#     --seed 42 \