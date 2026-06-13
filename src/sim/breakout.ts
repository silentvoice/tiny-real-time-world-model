export const FRAME_WIDTH = 64;
export const FRAME_HEIGHT = 64;
export const RGB_CHANNELS = 3;
export const FRAME_PIXELS = FRAME_WIDTH * FRAME_HEIGHT;
export const RGB_SIZE = FRAME_PIXELS * RGB_CHANNELS;
export const CONTEXT_FRAMES = 4;

export type RgbFrame = Uint8ClampedArray;
export type Action = 0 | 1 | 2;

export const ACTION_LABELS: Record<Action, string> = {
  0: "noop",
  1: "left",
  2: "right"
};

const BRICK_ROWS = 6;
const BRICK_COLS = 12;
const BRICK_W = 4;
const BRICK_H = 2;
const BRICK_X0 = 4;
const BRICK_Y0 = 8;
const BRICK_GAP_X = 1;
const BRICK_GAP_Y = 2;
const PADDLE_Y = 57;
const PADDLE_W = 12;
const PADDLE_H = 2;

const BRICK_COLORS = [
  [251, 92, 92],
  [255, 178, 76],
  [250, 225, 87],
  [87, 217, 139],
  [73, 190, 255],
  [166, 129, 255]
] as const;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function fill(frame: RgbFrame, color: readonly [number, number, number]) {
  for (let i = 0; i < RGB_SIZE; i += 3) {
    frame[i] = color[0];
    frame[i + 1] = color[1];
    frame[i + 2] = color[2];
  }
}

function setPixel(frame: RgbFrame, x: number, y: number, color: readonly [number, number, number]) {
  if (x < 0 || x >= FRAME_WIDTH || y < 0 || y >= FRAME_HEIGHT) return;
  const offset = (y * FRAME_WIDTH + x) * 3;
  frame[offset] = color[0];
  frame[offset + 1] = color[1];
  frame[offset + 2] = color[2];
}

function rect(
  frame: RgbFrame,
  x: number,
  y: number,
  w: number,
  h: number,
  color: readonly [number, number, number]
) {
  const x0 = Math.max(0, Math.floor(x));
  const y0 = Math.max(0, Math.floor(y));
  const x1 = Math.min(FRAME_WIDTH, Math.ceil(x + w));
  const y1 = Math.min(FRAME_HEIGHT, Math.ceil(y + h));
  for (let yy = y0; yy < y1; yy++) {
    for (let xx = x0; xx < x1; xx++) {
      setPixel(frame, xx, yy, color);
    }
  }
}

export class BreakoutSim {
  bricks: boolean[] = [];
  paddleX = 26;
  ballX = 32;
  ballY = 45;
  ballVx = 0.7;
  ballVy = -0.9;
  score = 0;
  lives = 3;
  ticks = 0;

  constructor() {
    this.reset();
  }

  reset() {
    this.bricks = new Array(BRICK_ROWS * BRICK_COLS).fill(true);
    this.paddleX = 26;
    this.ballX = 32;
    this.ballY = 45;
    this.ballVx = Math.random() > 0.5 ? 0.72 : -0.72;
    this.ballVy = -0.92;
    this.score = 0;
    this.lives = 3;
    this.ticks = 0;
  }

  step(action: Action) {
    this.ticks += 1;
    if (action === 1) this.paddleX -= 2.4;
    if (action === 2) this.paddleX += 2.4;
    this.paddleX = clamp(this.paddleX, 2, FRAME_WIDTH - PADDLE_W - 2);

    this.ballX += this.ballVx;
    this.ballY += this.ballVy;

    if (this.ballX <= 2 || this.ballX >= FRAME_WIDTH - 3) {
      this.ballVx *= -1;
      this.ballX = clamp(this.ballX, 2, FRAME_WIDTH - 3);
    }
    if (this.ballY <= 4) {
      this.ballVy = Math.abs(this.ballVy);
      this.ballY = 4;
    }

    const paddleHit =
      this.ballY >= PADDLE_Y - 1 &&
      this.ballY <= PADDLE_Y + PADDLE_H &&
      this.ballX >= this.paddleX - 1 &&
      this.ballX <= this.paddleX + PADDLE_W + 1 &&
      this.ballVy > 0;

    if (paddleHit) {
      const rel = (this.ballX - (this.paddleX + PADDLE_W / 2)) / (PADDLE_W / 2);
      this.ballVx = clamp(rel * 1.15, -1.2, 1.2);
      this.ballVy = -Math.max(0.8, Math.abs(this.ballVy));
      this.ballY = PADDLE_Y - 2;
    }

    for (let row = 0; row < BRICK_ROWS; row++) {
      for (let col = 0; col < BRICK_COLS; col++) {
        const idx = row * BRICK_COLS + col;
        if (!this.bricks[idx]) continue;
        const bx = BRICK_X0 + col * (BRICK_W + BRICK_GAP_X);
        const by = BRICK_Y0 + row * (BRICK_H + BRICK_GAP_Y);
        const hit =
          this.ballX >= bx - 1 &&
          this.ballX <= bx + BRICK_W &&
          this.ballY >= by - 1 &&
          this.ballY <= by + BRICK_H;
        if (hit) {
          this.bricks[idx] = false;
          this.ballVy *= -1;
          this.score += 10;
          row = BRICK_ROWS;
          break;
        }
      }
    }

    if (this.ballY > FRAME_HEIGHT + 2) {
      this.lives -= 1;
      this.ballX = this.paddleX + PADDLE_W / 2;
      this.ballY = 45;
      this.ballVx = Math.random() > 0.5 ? 0.72 : -0.72;
      this.ballVy = -0.92;
      if (this.lives <= 0) this.reset();
    }

    if (this.bricks.every((brick) => !brick)) {
      this.bricks.fill(true);
      this.ballVy = -Math.abs(this.ballVy);
    }
  }

  renderRgb(): RgbFrame {
    const frame = new Uint8ClampedArray(RGB_SIZE);
    fill(frame, [8, 10, 18]);

    rect(frame, 0, 0, FRAME_WIDTH, 3, [24, 30, 48]);
    rect(frame, 0, 0, 2, FRAME_HEIGHT, [18, 23, 38]);
    rect(frame, FRAME_WIDTH - 2, 0, 2, FRAME_HEIGHT, [18, 23, 38]);

    for (let row = 0; row < BRICK_ROWS; row++) {
      for (let col = 0; col < BRICK_COLS; col++) {
        const idx = row * BRICK_COLS + col;
        if (!this.bricks[idx]) continue;
        const bx = BRICK_X0 + col * (BRICK_W + BRICK_GAP_X);
        const by = BRICK_Y0 + row * (BRICK_H + BRICK_GAP_Y);
        rect(frame, bx, by, BRICK_W, BRICK_H, BRICK_COLORS[row]);
      }
    }

    rect(frame, this.paddleX, PADDLE_Y, PADDLE_W, PADDLE_H, [235, 241, 252]);
    rect(frame, this.paddleX + 1, PADDLE_Y - 1, PADDLE_W - 2, 1, [112, 221, 255]);

    const bx = Math.round(this.ballX);
    const by = Math.round(this.ballY);
    rect(frame, bx - 1, by - 1, 3, 3, [255, 255, 255]);
    setPixel(frame, bx, by, [255, 221, 103]);

    return frame;
  }
}

export function rgbToImageData(frame: RgbFrame): ImageData {
  const rgba = new Uint8ClampedArray(FRAME_PIXELS * 4);
  for (let src = 0, dst = 0; src < RGB_SIZE; src += 3, dst += 4) {
    rgba[dst] = frame[src];
    rgba[dst + 1] = frame[src + 1];
    rgba[dst + 2] = frame[src + 2];
    rgba[dst + 3] = 255;
  }
  return new ImageData(rgba, FRAME_WIDTH, FRAME_HEIGHT);
}

export function drawFrame(canvas: HTMLCanvasElement, frame: RgbFrame) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const scratch = document.createElement("canvas");
  scratch.width = FRAME_WIDTH;
  scratch.height = FRAME_HEIGHT;
  scratch.getContext("2d")?.putImageData(rgbToImageData(frame), 0, 0);

  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(scratch, 0, 0, canvas.width, canvas.height);
}

export function seedContext(sim: BreakoutSim): RgbFrame[] {
  const frames: RgbFrame[] = [];
  for (let i = 0; i < CONTEXT_FRAMES; i++) {
    sim.step(0);
    frames.push(sim.renderRgb());
  }
  return frames;
}

