# Use the NVIDIA Triton Inference Server as the base image
# FROM nvcr.io/nvidia/tritonserver:24.07-py3
FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

COPY . /mnt
WORKDIR /mnt

# Install necessary dependencies
# RUN [ "$(uname -m)" != "x86_64" ] && arch="sbsa" || arch="x86_64" \
#     && curl -o /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/$arch/cuda-keyring_1.0-1_all.deb \
#     && apt install /tmp/cuda-keyring.deb \
#     && rm /tmp/cuda-keyring.deb \
#     && apt-get update -q=2

# RUN apt-get update && apt-get install -y \
#     openmpi-bin

# RUN pip3 install --extra-index-url https://pypi.nvidia.com tensorrt-llm==0.11.0
# Install Hugging Face libraries
RUN pip3 install huggingface_hub
RUN pip3 install SentencePiece
# RUN pip3 insta

# Environment variables
ENV UNIFIED_CKPT_PATH=/mnt/ckpt/phi2/
ENV ENGINE_PATH=/mnt/engines/phi2/

# Create directories
RUN mkdir -p $HF_PHI_MODEL $UNIFIED_CKPT_PATH $ENGINE_PATH

# Download the model from Hugging Face and set the HF_PHI_MODEL environment variable
# Download the model from Hugging Face and set the HF_PHI_MODEL environment variable
RUN HF_PHI_MODEL=$(python3 -c "from pathlib import Path; from huggingface_hub import snapshot_download; print(Path(snapshot_download('microsoft/phi-2')))") && \
    echo "HF_PHI_MODEL=${HF_PHI_MODEL}" >> /mnt/.env

# Source the environment file to make the variable available
# RUN . /etc/environment && echo $HF_PHI_MODEL
# Copy your scripts and other necessary files into the Docker image
# COPY phi /mnt/phi
# COPY tools /mnt/tools
# COPY scripts /mnt/scripts
# COPY phi_ifb /mnt/phi_ifb

# Convert the checkpoint
RUN cp -r /mnt/all_models/inflight_batcher_llm/ /mnt/phi_ifb

# Fill in the configuration templates (uncomment and adjust if needed)
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_PHI_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_PHI_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/ensemble/config.pbtxt triton_max_batch_size:64
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

# Expose port 8000
EXPOSE 8000

CMD ["/mnt/entrypoint.sh"]
# RUN . /etc/environment && echo $HF_PHI_MODEL && python3 phi/convert_checkpoint.py --model_dir ${HF_PHI_MODEL} \
#                                            --output_dir ${UNIFIED_CKPT_PATH} \
#                                            --dtype float16

# CMD ["python3", "phi/convert_checkpoint.py", "--model_dir", "echo ${HF_PHI_MODEL}", "--output_dir", "echo ${UNIFIED_CKPT_PATH}", "--dtype", "float16"]

# Build the engine
# RUN trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} \
#                  --remove_input_padding enable \
#                  --gpt_attention_plugin float16 \
#                  --context_fmha enable \
#                  --gemm_plugin float16 \
#                  --output_dir ${ENGINE_PATH} \
#                  --use_custom_all_reduce disable \
#                  --paged_kv_cache enable \
#                  --max_batch_size 64

# CMD [".", "/etc/environment", "&&", "trtllm-build", "--checkpoint_dir", "${UNIFIED_CKPT_PATH}", "--remove_input_padding", "enable", "--gpt_attention_plugin", "float16", "--context_fmha", "enable", "--gemm_plugin", "float16", "--output_dir", "${ENGINE_PATH}", "--use_custom_all_reduce", "disable", "--paged_kv_cache", "enable", "--max_batch_size", "64"]

# Command to run the Triton server
# CMD ["python3", "/mnt/scripts/launch_triton_server.py", "--world_size", "1", "--model_repo=/mnt/phi_ifb/"]


# CMD ["bash", "-c", "python3 scripts/launch_triton_server.py --world_size 1 --model_repo=phi_ifb/"]
