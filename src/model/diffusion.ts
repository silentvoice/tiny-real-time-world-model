import * as ort from "onnxruntime-web/webgpu";
import {
  ACTION_LABELS,
  type Action,
  CONTEXT_FRAMES,
  FRAME_HEIGHT,
  FRAME_WIDTH,
  RGB_SIZE,
  type RgbFrame
} from "../sim/breakout";

const INPUT_CHANNELS = 3 + CONTEXT_FRAMES * 3 + 3 + 1;
const SIGMA_MIN = 0.02;

function normalSample() {
  const u = Math.max(Number.MIN_VALUE, Math.random());
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function rgbToFloat(frame: RgbFrame, out: Float32Array, channelOffset: number) {
  const plane = FRAME_WIDTH * FRAME_HEIGHT;
  for (let p = 0; p < plane; p++) {
    out[(channelOffset + 0) * plane + p] = frame[p * 3] / 127.5 - 1;
    out[(channelOffset + 1) * plane + p] = frame[p * 3 + 1] / 127.5 - 1;
    out[(channelOffset + 2) * plane + p] = frame[p * 3 + 2] / 127.5 - 1;
  }
}

function rgbToModelFrame(frame: RgbFrame): Float32Array {
  const out = new Float32Array(3 * FRAME_WIDTH * FRAME_HEIGHT);
  rgbToFloat(frame, out, 0);
  return out;
}

function floatToRgb(frame: Float32Array): RgbFrame {
  const out = new Uint8ClampedArray(RGB_SIZE);
  const plane = FRAME_WIDTH * FRAME_HEIGHT;
  for (let p = 0; p < plane; p++) {
    out[p * 3] = Math.round((Math.max(-1, Math.min(1, frame[p])) + 1) * 127.5);
    out[p * 3 + 1] = Math.round((Math.max(-1, Math.min(1, frame[plane + p])) + 1) * 127.5);
    out[p * 3 + 2] = Math.round((Math.max(-1, Math.min(1, frame[2 * plane + p])) + 1) * 127.5);
  }
  return out;
}

function makeInput(
  context: RgbFrame[],
  action: Action,
  noisy: Float32Array,
  sigma: number
): Float32Array {
  const plane = FRAME_WIDTH * FRAME_HEIGHT;
  const input = new Float32Array(INPUT_CHANNELS * plane);

  input.set(noisy, 0);

  let channel = 3;
  for (const frame of context) {
    rgbToFloat(frame, input, channel);
    channel += 3;
  }

  for (let a = 0; a < 3; a++) {
    input.fill(a === action ? 1 : 0, channel * plane, (channel + 1) * plane);
    channel += 1;
  }

  input.fill(sigma, channel * plane, (channel + 1) * plane);
  return input;
}

function makeSchedule(steps: number, maxSigma: number): number[] {
  const count = Math.max(1, steps);
  if (count === 1) return [maxSigma, 0];
  const sigmas = [];
  for (let i = 0; i < count; i++) {
    const t = i / (count - 1);
    sigmas.push(maxSigma * Math.pow(SIGMA_MIN / maxSigma, t));
  }
  sigmas.push(0);
  return sigmas;
}

export class DiffusionSampler {
  private session: ort.InferenceSession;

  private constructor(session: ort.InferenceSession) {
    this.session = session;
  }

  static async create(modelBytes: Uint8Array) {
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["webgpu", "wasm"],
      graphOptimizationLevel: "all"
    });
    return new DiffusionSampler(session);
  }

  async sample(
    context: RgbFrame[],
    action: Action,
    denoiseSteps: number,
    maxSigma: number
  ): Promise<RgbFrame> {
    const plane = FRAME_WIDTH * FRAME_HEIGHT;
    let current = rgbToModelFrame(context[context.length - 1]);
    for (let i = 0; i < current.length; i++) {
      current[i] += normalSample() * maxSigma;
    }

    const schedule = makeSchedule(denoiseSteps, maxSigma);
    for (let i = 0; i < schedule.length - 1; i++) {
      const sigma = schedule[i];
      const nextSigma = schedule[i + 1];
      const input = makeInput(context, action, current, sigma);
      const feeds = {
        input: new ort.Tensor("float32", input, [1, INPUT_CHANNELS, FRAME_HEIGHT, FRAME_WIDTH])
      };
      const outputMap = await this.session.run(feeds);
      const outputName = this.session.outputNames[0] ?? "pred";
      const pred = outputMap[outputName].data as Float32Array;

      if (nextSigma === 0) {
        current = new Float32Array(pred);
      } else {
        const updated = new Float32Array(current.length);
        const ratio = nextSigma / Math.max(SIGMA_MIN, sigma);
        for (let j = 0; j < current.length; j++) {
          updated[j] = pred[j] + ratio * (current[j] - pred[j]);
        }
        current = updated;
      }
    }

    return floatToRgb(current);
  }
}

export function describeAction(action: Action) {
  return ACTION_LABELS[action];
}
