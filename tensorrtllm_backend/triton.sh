#!/bin/bash
set -e

. /mnt/.env

echo "Running Phi checkpoint conversion"
python3 phi/convert_checkpoint.py --model_dir "${HF_PHI_MODEL}" --output_dir "${UNIFIED_CKPT_PATH}" --dtype float16

echo "Building TensorRT engine"
trtllm-build --checkpoint_dir "${UNIFIED_CKPT_PATH}" \
             --remove_input_padding enable \
             --gpt_attention_plugin float16 \
             --context_fmha enable \
             --gemm_plugin float16 \
             --output_dir "${ENGINE_PATH}" \
             --use_custom_all_reduce disable \
             --paged_kv_cache enable \
             --max_batch_size 64

echo "Launching Triton server"
python3 scripts/launch_triton_server.py --world_size 1 --model_repo=phi_ifb/
tail -f /dev/null
