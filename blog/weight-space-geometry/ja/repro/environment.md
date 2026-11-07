# Environment record

Artifact provenance: upstream commit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`; model revision `2fc06364715b967f1860aea9cf38778875588b17`; edit SHA-256 `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`.

The reference measurements used Hugging Face Transformers 5.13.1 and PyTorch 2.13.0 with CUDA 13.0. The generation pilot used vLLM 0.27.1 with Transformers 5.9.0 and PyTorch 2.13.0. The task-local FlashInfer Python package and cubin were both 0.6.11.post2; do not mix versions.

For the Qwen3.5 generation path, the validated settings were `VLLM_USE_FLASHINFER_SAMPLER=0`, `TRITON_ATTN`, Triton GDN prefill, disabled FlashInfer autotuning, and disabled CuteDSL warmup. This is a record of one environment, not a portable lockfile. GPU availability, CUDA driver, and local model storage remain prerequisites.

The public edit and its verification metadata are downloaded by `prepare.py`; model weights and benchmark data are intentionally not included.
