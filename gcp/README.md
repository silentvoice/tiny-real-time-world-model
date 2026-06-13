# GCP Training

These scripts run training on a Compute Engine GPU VM with an attached service
account. No service account key is created or stored locally.

## Bootstrap cloud resources

```bash
PROJECT_ID="your-project-id" \
REGION="us-central1" \
SERVICE_ACCOUNT_NAME="tiny-world-model-runner" \
BUCKET_NAME="your-project-id-tiny-world-model" \
bash gcp/bootstrap.sh
```

The bootstrap script enables required APIs, creates a service account if needed,
grants storage/logging permissions, and creates a GCS bucket for model artifacts.
Set `SERVICE_ACCOUNT_PROJECT_ROLE` only when your own sandbox requires broader
project-level permissions.

## Launch a training VM

Push the repository to GitHub first, then run:

```bash
PROJECT_ID="your-project-id" \
ZONE="us-central1-a" \
SERVICE_ACCOUNT_EMAIL="tiny-world-model-runner@your-project-id.iam.gserviceaccount.com" \
BUCKET_NAME="your-project-id-tiny-world-model" \
REPO_URL="https://github.com/your-user/tiny-real-time-world-model.git" \
TRAIN_STEPS="40000" \
MACHINE_TYPE="g2-standard-8" \
ACCELERATOR="type=nvidia-l4,count=1" \
bash gcp/create-training-vm.sh
```

The VM startup script clones the repo, installs Python dependencies, trains the
denoiser, exports `tiny_denoiser.onnx`, and uploads artifacts to GCS.

## Retrieve the model

After training completes:

```bash
gcloud storage cp \
  gs://your-project-id-tiny-world-model/latest/tiny_denoiser.onnx \
  public/model/tiny_denoiser.onnx
```

Then build and publish the GitHub Pages app.

## Cost controls

- Use Spot provisioning for training VMs.
- Checkpoint frequently.
- Stop or delete the VM after training.
- Keep model artifacts in GCS; do not commit large checkpoints.
