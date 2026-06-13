#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${ZONE:=us-central1-a}"
: "${INSTANCE_NAME:=tiny-world-model-train}"
: "${SERVICE_ACCOUNT_EMAIL:?Set SERVICE_ACCOUNT_EMAIL}"
: "${BUCKET_NAME:?Set BUCKET_NAME}"
: "${REPO_URL:?Set REPO_URL}"
: "${TRAIN_STEPS:=40000}"
: "${BATCH_SIZE:=128}"
: "${MODEL_WIDTH:=96}"
: "${MODEL_BLOCKS:=14}"
: "${MACHINE_TYPE:=g2-standard-8}"
: "${ACCELERATOR:=type=nvidia-l4,count=1}"
: "${BOOT_DISK_SIZE:=200GB}"

STARTUP_SCRIPT="$(pwd)/gcp/startup-train.sh"

gcloud compute instances create "${INSTANCE_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator "${ACCELERATOR}" \
  --maintenance-policy TERMINATE \
  --provisioning-model SPOT \
  --instance-termination-action STOP \
  --image-family pytorch-latest-gpu \
  --image-project deeplearning-platform-release \
  --boot-disk-size "${BOOT_DISK_SIZE}" \
  --boot-disk-type pd-balanced \
  --service-account "${SERVICE_ACCOUNT_EMAIL}" \
  --scopes cloud-platform \
  --metadata-from-file startup-script="${STARTUP_SCRIPT}" \
  --metadata \
repo-url="${REPO_URL}",bucket-name="${BUCKET_NAME}",train-steps="${TRAIN_STEPS}",batch-size="${BATCH_SIZE}",model-width="${MODEL_WIDTH}",model-blocks="${MODEL_BLOCKS}"

echo "created ${INSTANCE_NAME}"
echo "tail logs with:"
echo "gcloud compute ssh ${INSTANCE_NAME} --project ${PROJECT_ID} --zone ${ZONE} --command 'tail -f /var/log/startup-script.log'"

