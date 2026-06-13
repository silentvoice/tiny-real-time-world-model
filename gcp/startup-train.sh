#!/usr/bin/env bash
set -euo pipefail

metadata() {
  curl -fsS -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

REPO_URL="$(metadata repo-url)"
BUCKET_NAME="$(metadata bucket-name)"
TRAIN_STEPS="$(metadata train-steps)"
BATCH_SIZE="$(metadata batch-size)"
MODEL_WIDTH="$(metadata model-width)"
MODEL_BLOCKS="$(metadata model-blocks)"

WORKDIR="/opt/tiny-real-time-world-model"
CHECKPOINT_DIR="${WORKDIR}/checkpoints/tiny-denoiser"
MODEL_PATH="${WORKDIR}/public/model/tiny_denoiser.onnx"

echo "repo=${REPO_URL}"
echo "bucket=gs://${BUCKET_NAME}"
echo "steps=${TRAIN_STEPS}"

sudo rm -rf "${WORKDIR}"
sudo git clone "${REPO_URL}" "${WORKDIR}"
sudo chown -R "$(id -u):$(id -g)" "${WORKDIR}"
cd "${WORKDIR}"

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[eval]"

python -m world_model_lab.train \
  --steps "${TRAIN_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --width "${MODEL_WIDTH}" \
  --blocks "${MODEL_BLOCKS}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --save-every 1000 \
  --sample-every 1000

python -m world_model_lab.export_onnx \
  --checkpoint "${CHECKPOINT_DIR}/last.pt" \
  --output "${MODEL_PATH}"

python -m world_model_lab.eval_rollout \
  --checkpoint "${CHECKPOINT_DIR}/last.pt" \
  --output "${WORKDIR}/artifacts/rollout.gif" \
  --steps 160

gcloud storage cp "${MODEL_PATH}" "gs://${BUCKET_NAME}/latest/tiny_denoiser.onnx"
gcloud storage cp "${WORKDIR}/artifacts/rollout.gif" "gs://${BUCKET_NAME}/latest/rollout.gif"
gcloud storage cp "${CHECKPOINT_DIR}/last.pt" "gs://${BUCKET_NAME}/checkpoints/last.pt"
gcloud storage cp --recursive "${CHECKPOINT_DIR}/samples" "gs://${BUCKET_NAME}/samples"

echo "training complete"
echo "model: gs://${BUCKET_NAME}/latest/tiny_denoiser.onnx"

