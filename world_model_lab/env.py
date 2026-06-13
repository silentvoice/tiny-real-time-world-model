from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FRAME_WIDTH = 64
FRAME_HEIGHT = 64
CONTEXT_FRAMES = 4
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3

BRICK_ROWS = 6
BRICK_COLS = 12
BRICK_W = 4
BRICK_H = 2
BRICK_X0 = 4
BRICK_Y0 = 8
BRICK_GAP_X = 1
BRICK_GAP_Y = 2
PADDLE_Y = 57
PADDLE_W = 12
PADDLE_H = 2

BRICK_COLORS = np.array(
    [
        [251, 92, 92],
        [255, 178, 76],
        [250, 225, 87],
        [87, 217, 139],
        [73, 190, 255],
        [166, 129, 255],
    ],
    dtype=np.uint8,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def _rect(frame: np.ndarray, x: float, y: float, w: float, h: float, color: np.ndarray | tuple[int, int, int]):
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(FRAME_WIDTH, int(np.ceil(x + w)))
    y1 = min(FRAME_HEIGHT, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1, :] = color


@dataclass
class BreakoutEnv:
    """Small deterministic Breakout-like environment rendered directly to 64x64 RGB."""

    seed: int | None = None
    rng: np.random.Generator = field(init=False)
    bricks: np.ndarray = field(init=False)
    paddle_x: float = field(init=False, default=26.0)
    ball_x: float = field(init=False, default=32.0)
    ball_y: float = field(init=False, default=45.0)
    ball_vx: float = field(init=False, default=0.72)
    ball_vy: float = field(init=False, default=-0.92)
    score: int = field(init=False, default=0)
    lives: int = field(init=False, default=3)
    ticks: int = field(init=False, default=0)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self):
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=np.bool_)
        self.paddle_x = 26.0
        self.ball_x = 32.0
        self.ball_y = 45.0
        self.ball_vx = 0.72 if self.rng.random() > 0.5 else -0.72
        self.ball_vy = -0.92
        self.score = 0
        self.lives = 3
        self.ticks = 0

    def scripted_action(self, random_rate: float = 0.18) -> int:
        if self.rng.random() < random_rate:
            return int(self.rng.integers(0, NUM_ACTIONS))
        target = self.ball_x - PADDLE_W / 2
        if target < self.paddle_x - 1.5:
            return ACTION_LEFT
        if target > self.paddle_x + 1.5:
            return ACTION_RIGHT
        return ACTION_NOOP

    def step(self, action: int):
        self.ticks += 1
        if action == ACTION_LEFT:
            self.paddle_x -= 2.4
        elif action == ACTION_RIGHT:
            self.paddle_x += 2.4
        self.paddle_x = _clamp(self.paddle_x, 2, FRAME_WIDTH - PADDLE_W - 2)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_x <= 2 or self.ball_x >= FRAME_WIDTH - 3:
            self.ball_vx *= -1
            self.ball_x = _clamp(self.ball_x, 2, FRAME_WIDTH - 3)
        if self.ball_y <= 4:
            self.ball_vy = abs(self.ball_vy)
            self.ball_y = 4

        paddle_hit = (
            PADDLE_Y - 1 <= self.ball_y <= PADDLE_Y + PADDLE_H
            and self.paddle_x - 1 <= self.ball_x <= self.paddle_x + PADDLE_W + 1
            and self.ball_vy > 0
        )
        if paddle_hit:
            rel = (self.ball_x - (self.paddle_x + PADDLE_W / 2)) / (PADDLE_W / 2)
            self.ball_vx = _clamp(rel * 1.15, -1.2, 1.2)
            self.ball_vy = -max(0.8, abs(self.ball_vy))
            self.ball_y = PADDLE_Y - 2

        hit_done = False
        for row in range(BRICK_ROWS):
            if hit_done:
                break
            for col in range(BRICK_COLS):
                if not self.bricks[row, col]:
                    continue
                bx = BRICK_X0 + col * (BRICK_W + BRICK_GAP_X)
                by = BRICK_Y0 + row * (BRICK_H + BRICK_GAP_Y)
                hit = bx - 1 <= self.ball_x <= bx + BRICK_W and by - 1 <= self.ball_y <= by + BRICK_H
                if hit:
                    self.bricks[row, col] = False
                    self.ball_vy *= -1
                    self.score += 10
                    hit_done = True
                    break

        if self.ball_y > FRAME_HEIGHT + 2:
            self.lives -= 1
            self.ball_x = self.paddle_x + PADDLE_W / 2
            self.ball_y = 45.0
            self.ball_vx = 0.72 if self.rng.random() > 0.5 else -0.72
            self.ball_vy = -0.92
            if self.lives <= 0:
                self.reset()

        if not self.bricks.any():
            self.bricks[:, :] = True
            self.ball_vy = -abs(self.ball_vy)

    def render_rgb(self) -> np.ndarray:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:, :, :] = np.array([8, 10, 18], dtype=np.uint8)
        _rect(frame, 0, 0, FRAME_WIDTH, 3, (24, 30, 48))
        _rect(frame, 0, 0, 2, FRAME_HEIGHT, (18, 23, 38))
        _rect(frame, FRAME_WIDTH - 2, 0, 2, FRAME_HEIGHT, (18, 23, 38))

        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                if not self.bricks[row, col]:
                    continue
                bx = BRICK_X0 + col * (BRICK_W + BRICK_GAP_X)
                by = BRICK_Y0 + row * (BRICK_H + BRICK_GAP_Y)
                _rect(frame, bx, by, BRICK_W, BRICK_H, BRICK_COLORS[row])

        _rect(frame, self.paddle_x, PADDLE_Y, PADDLE_W, PADDLE_H, (235, 241, 252))
        _rect(frame, self.paddle_x + 1, PADDLE_Y - 1, PADDLE_W - 2, 1, (112, 221, 255))

        bx = round(self.ball_x)
        by = round(self.ball_y)
        _rect(frame, bx - 1, by - 1, 3, 3, (255, 255, 255))
        if 0 <= bx < FRAME_WIDTH and 0 <= by < FRAME_HEIGHT:
            frame[by, bx, :] = np.array([255, 221, 103], dtype=np.uint8)

        return frame


def make_context(env: BreakoutEnv) -> deque[np.ndarray]:
    context: deque[np.ndarray] = deque(maxlen=CONTEXT_FRAMES)
    for _ in range(CONTEXT_FRAMES):
        env.step(ACTION_NOOP)
        context.append(env.render_rgb())
    return context


def frame_to_chw(frame: np.ndarray) -> np.ndarray:
    return np.transpose(frame, (2, 0, 1))


def frames_to_context(frames: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate([frame_to_chw(frame) for frame in frames], axis=0)

