#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${REGION:=us-central1}"
: "${SERVICE_ACCOUNT_NAME:=tiny-world-model-runner}"
: "${BUCKET_NAME:=${PROJECT_ID}-tiny-world-model}"
: "${SERVICE_ACCOUNT_PROJECT_ROLE:=}"

SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
MEMBER="serviceAccount:${SERVICE_ACCOUNT_EMAIL}"

echo "project: ${PROJECT_ID}"
echo "service account: ${SERVICE_ACCOUNT_EMAIL}"
echo "bucket: gs://${BUCKET_NAME}"

gcloud services enable \
  compute.googleapis.com \
  iam.googleapis.com \
  storage.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  --project "${PROJECT_ID}"

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "Tiny world model trainer"
fi

for role in \
  roles/storage.admin \
  roles/logging.logWriter \
  roles/monitoring.metricWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "${MEMBER}" \
    --role "${role}" \
    --condition=None >/dev/null
done

if [[ -n "${SERVICE_ACCOUNT_PROJECT_ROLE}" ]]; then
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member "${MEMBER}" \
    --role "${SERVICE_ACCOUNT_PROJECT_ROLE}" \
    --condition=None >/dev/null
fi

if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project "${PROJECT_ID}" \
    --location "${REGION}" \
    --uniform-bucket-level-access
fi

echo "bootstrap complete"
