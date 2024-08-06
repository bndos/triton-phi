FROM nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3

COPY . /mnt
WORKDIR /mnt

RUN pip3 install huggingface_hub
RUN pip3 install SentencePiece

ENV UNIFIED_CKPT_PATH=/mnt/ckpt/phi2/
ENV ENGINE_PATH=/mnt/engines/phi2/

RUN mkdir -p $HF_PHI_MODEL $UNIFIED_CKPT_PATH $ENGINE_PATH

RUN HF_PHI_MODEL=$(python3 -c "from pathlib import Path; from huggingface_hub import snapshot_download; print(Path(snapshot_download('microsoft/phi-2')))") && \
    echo "HF_PHI_MODEL=${HF_PHI_MODEL}" >> /mnt/.env

RUN cp -r /mnt/all_models/inflight_batcher_llm/ /mnt/phi_ifb

RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/preprocessing/config.pbtxt tokenizer_dir:${HF_PHI_MODEL},triton_max_batch_size:64,preprocessing_instance_count:1
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/postprocessing/config.pbtxt tokenizer_dir:${HF_PHI_MODEL},triton_max_batch_size:64,postprocessing_instance_count:1
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:64,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/ensemble/config.pbtxt triton_max_batch_size:64
RUN . /mnt/.env && python3 /mnt/tools/fill_template.py -i /mnt/phi_ifb/tensorrt_llm/config.pbtxt triton_backend:tensorrtllm,triton_max_batch_size:64,decoupled_mode:False,max_beam_width:1,engine_dir:${ENGINE_PATH},max_tokens_in_paged_kv_cache:2560,max_attention_window_size:2560,kv_cache_free_gpu_mem_fraction:0.5,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0

EXPOSE 8000

CMD ["/mnt/triton.sh"]
