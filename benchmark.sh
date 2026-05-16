# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/GEdit-Bench \
#     --edit_file /data/datasets/GEdit-Bench/edit.json \
#     --data_root /data/datasets/GEdit-Bench

# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/MagicBrush-test \
#     --edit_file /data/datasets/MagicBrush-test/edit.json \
#     --data_root /data/datasets/MagicBrush-test

# CUDA_VISIBLE_DEVICES=1 python scripts/benchmark_moe.py \
#     --output-dir /data/code/models/0-output/emu_edit_test_set \
#     --edit_file /data/datasets/emu_edit_test_set/edit.json \
#     --data_root /data/datasets/emu_edit_test_set

CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_moe.py \
    --output-dir /data/code/models/0-output/AnyEdit \
    --edit_file /data/datasets/AnyEdit/edit.json \
    --data_root /data/datasets/AnyEdit