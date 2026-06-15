# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/GEdit-Bench \
#     --edit-file /data/datasets/GEdit-Bench/edit.json \
#     --data-root /data/datasets/GEdit-Bench

# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/MagicBrush-test \
#     --edit-file /data/datasets/MagicBrush-test/edit.json \
#     --data-root /data/datasets/MagicBrush-test

# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/emu_edit_test_set \
#     --edit-file /data/datasets/emu_edit_test_set/edit.json \
#     --data-root /data/datasets/emu_edit_test_set

CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
    --top-k 1 \
    --lora-path train/runs/train_route+rl+aux+token_subset/20260613-115237/ckpt/5000 \
    --output-dir /data/repos/models/0-output/AnyEdit-Test/ \
    --edit-file /data/datasets/AnyEdit/edit.json \
    --saved-suffix-model +GRPO_0.1tokens \
    --data-root /data/datasets/AnyEdit 2>&1 > benchmark_rl_sign.log &