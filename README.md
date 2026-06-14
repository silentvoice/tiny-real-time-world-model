# Tiny Real-Time World Model

A browser-playable diffusion world model trained on a small Breakout-like game.

The project has two parts:

- a deterministic pixel game used to generate unlimited training data
- a tiny action-conditioned denoising diffusion model that predicts the next
  frame from recent frames and keyboard input

The demo runs as a static web app and can be hosted on GitHub Pages. When a
trained ONNX model is present, the browser uses ONNX Runtime Web with WebGPU or
WASM to predict a live neural dream layer while the game remains playable.

Live demo: <https://silentvoice.github.io/tiny-real-time-world-model/>

## Demo modes

- **Real simulator**: normal Breakout-like game physics in JavaScript.
- **Neural world**: the game remains responsive, and the exported diffusion
  model continuously predicts a noisy next-frame layer from the last four frames
  and the current action.

Neural mode is intentionally imperfect. The tiny model still drifts and invents
artifacts, but controls, score, and collisions stay live so the demo feels like a
game instead of a slow autonomous rollout.

## Quick start

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. The simulator works immediately. Neural mode
is enabled by the included model at:

```text
public/model/tiny_denoiser.onnx
```

The checked-in model is intentionally small enough to load as a normal static
asset. It was trained on the included simulator for a compact browser demo, not
for benchmark-quality rollouts.

## Train a model

Install the Python package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[eval]"
```

Run a small local smoke train:

```bash
python -m world_model_lab.train --steps 200 --batch-size 16 --checkpoint-dir checkpoints/smoke
python -m world_model_lab.export_onnx --checkpoint checkpoints/smoke/last.pt --output public/model/tiny_denoiser.onnx
```

For a useful browser model, train on a GPU:

```bash
python -m world_model_lab.train \
  --steps 5000 \
  --batch-size 128 \
  --width 96 \
  --blocks 14 \
  --checkpoint-dir checkpoints/tiny-denoiser
python -m world_model_lab.export_onnx \
  --checkpoint checkpoints/tiny-denoiser/last.pt \
  --output public/model/tiny_denoiser.onnx
```

## How it works

Each training sample contains:

```text
context: 4 RGB frames, 64x64
action: noop, left, or right
target: the next RGB frame
```

During training, Gaussian noise is added to the target frame. The model receives
the noisy target, the context frames, the action encoded as image planes, and a
noise-level plane. It learns to predict the clean next frame.

At inference time, the browser warm-starts from the latest context frame plus
noise and runs a short deterministic denoising schedule. The app blends that
prediction over the live simulator so keyboard control stays responsive even
when model inference is slow.

## Repository layout

```text
src/                 Browser demo
world_model_lab/     Python simulator, training, and ONNX export
gcp/                 GCP bootstrap and training helper scripts
public/model/        Optional exported ONNX model for GitHub Pages
```

## GCP training

The GCP scripts are designed for a VM with an attached service account. They do
not require service account keys and should not write credentials into the repo.

See [gcp/README.md](gcp/README.md).

## License

MIT
