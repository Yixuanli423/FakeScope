#!/usr/bin/env bash
# Authors' FakeScope launcher, adapted from Q-Instruct's "after" strategy.
# Keep backbone runtimes in separate environments; see docs/training.md.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${LLAVA_ROOT:?Set LLAVA_ROOT to the installed backbone checkout directory.}"
: "${DATA_PATH:?Set DATA_PATH to the training annotation JSON.}"
: "${IMAGE_FOLDER:?Set IMAGE_FOLDER to the training image directory.}"
training_entry="${LLAVA_ROOT}/llava/train/train_mem.py"
deepspeed_config="${DEEPSPEED_CONFIG:-${LLAVA_ROOT}/scripts/zero3.json}"
for required_file in "$training_entry" "$deepspeed_config" "$DATA_PATH"; do
    if [[ ! -f "$required_file" ]]; then
        printf 'Required file not found: %s\n' "$required_file" >&2
        exit 1
    fi
done
if [[ ! -d "$IMAGE_FOLDER" ]]; then
    printf 'Image directory not found: %s\n' "$IMAGE_FOLDER" >&2
    exit 1
fi
launcher_args=(--master_port "${MASTER_PORT:-29500}")
if [[ -n "${NUM_GPUS:-}" ]]; then
    launcher_args+=(--num_gpus "$NUM_GPUS")
fi
dry_run=false
extra_args=()
for argument in "$@"; do
    if [[ "$argument" == "--dry-run" ]]; then
        dry_run=true
    else
        extra_args+=("$argument")
    fi
done
command=(deepspeed "${launcher_args[@]}" "$training_entry" \
    --deepspeed "$deepspeed_config" \
    --model_name_or_path "${MODEL_PATH:-liuhaotian/llava-v1.5-7b}" \
    --version v1 \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR:-${repo_root}/checkpoints/llava-v1.5-7b-fakeinstruct}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "${REPORT_TO:-wandb}" \
    ${extra_args[@]+"${extra_args[@]}"})
if [[ "$dry_run" == true ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    exec "${command[@]}"
fi
