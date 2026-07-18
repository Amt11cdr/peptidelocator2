# PeptideLocator2 — Docker image
# Runs the Gradio demo on port 7860.
#
# Build (CPU-only, for lab webserver without GPU):
#   docker build -t peptidelocator2 .
#
# Build (GPU, if the host has CUDA):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 \
#                -t peptidelocator2 .
#
# Run:
#   docker run -p 7860:7860 peptidelocator2
#
# With fine-tuned weights (optional):
#   docker run -p 7860:7860 \
#     -v /path/to/checkpoints:/app/checkpoints \
#     -e FINETUNE_SITES_PATH=/app/checkpoints/finetune_esm2_8m_sites_fold0_seed0 \
#     -e FINETUNE_PEPTIDES_PATH=/app/checkpoints/finetune_esm2_8m_peptides_fold0_seed0 \
#     peptidelocator2

FROM python:3.10-slim

# Build arg: override to use GPU wheels
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first (CPU by default, smaller image)
RUN pip install --no-cache-dir \
    torch \
    --index-url ${TORCH_INDEX}

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn tqdm pyyaml pyarrow scipy \
    "transformers>=4.35.0" \
    "gradio>=4.0.0" \
    plotly \
    sentencepiece

# Pre-download ESM2-8M at build time so the container works offline
RUN python -c "\
from transformers import EsmModel, EsmTokenizer; \
print('Downloading ESM2-8M tokenizer...'); \
EsmTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
print('Downloading ESM2-8M model...'); \
EsmModel.from_pretrained('facebook/esm2_t6_8M_UR50D'); \
print('Done.')"

# Copy application code
COPY app.py .
COPY peptidelocator/ peptidelocator/

# Copy MLP head weights if they exist (built by slurm_save_model.sh)
# If missing, the app falls back to untrained heads and labels them clearly.
RUN mkdir -p models
COPY models/ models/

# Expose Gradio port
EXPOSE 7860

# Gradio env vars
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
# Suppress HuggingFace telemetry
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

CMD ["python", "app.py"]
