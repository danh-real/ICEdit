LORA=train/runs/train_route_rl_sign/20260608-052753/ckpt/10000

CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
    --instruction "remove the bowl of fruit, then add a smaller version of the bowl of fruit" \
    --seed 42 \
    --lora-path $LORA \
    --output-suffix "decomposedResize" \

CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
    --instruction "zoom out the bowl of fruit" \
    --seed 42 \
    --lora-path $LORA \
    --output-suffix "resize" \

CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe.py \
    --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
    --instruction "remove the bowl of fruit" \
    --seed 42 \
    --lora-path $LORA \
    --output-suffix "remove" \

# CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe_step1.py \
#     --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
#     --instruction "remove the bowl of fruit" \
#     --seed 42 \
#     --lora-path $LORA/pytorch_lora_weights.safetensors \

# CUDA_VISIBLE_DEVICES=1 python scripts/inference_moe_step2.py \
#     --image /data/datasets/AnyEdit/anybench/resize/input_img/COCO_train2014_000000320451.jpg \
#     --step1-image COCO_train2014_000000320451.jpg_remove.jpg \
#     --instruction "add one smaller version of the bowl of fruit from the left" \
#     --seed 42 \
#     --lora-path $LORA/pytorch_lora_weights.safetensors \

# COCO_train2014_000000026928