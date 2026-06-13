import { Github, Pause, Play, RotateCcw, Sparkles } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DiffusionSampler, describeAction } from "./model/diffusion";
import {
  type Action,
  BreakoutSim,
  CONTEXT_FRAMES,
  drawFrame,
  type RgbFrame,
  seedContext
} from "./sim/breakout";

type Mode = "sim" | "neural";
type ModelState = "loading" | "ready" | "missing" | "error";

const MODEL_URL = `${import.meta.env.BASE_URL}model/tiny_denoiser.onnx`;

function looksLikeOnnx(buffer: ArrayBuffer, contentType: string | null) {
  if (buffer.byteLength < 1024) return false;
  if (contentType?.includes("text/html")) return false;
  const head = new TextDecoder().decode(new Uint8Array(buffer.slice(0, 32))).toLowerCase();
  return !head.includes("<!doctype") && !head.includes("<html");
}

function currentAction(keys: Set<string>): Action {
  if (keys.has("ArrowLeft") || keys.has("KeyA")) return 1;
  if (keys.has("ArrowRight") || keys.has("KeyD")) return 2;
  return 0;
}

function pushContext(context: RgbFrame[], frame: RgbFrame) {
  const next = context.slice(-CONTEXT_FRAMES + 1);
  next.push(frame);
  return next;
}

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const thumbRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const simRef = useRef(new BreakoutSim());
  const contextRef = useRef<RgbFrame[]>(seedContext(simRef.current));
  const samplerRef = useRef<DiffusionSampler | null>(null);
  const keysRef = useRef(new Set<string>());
  const runningRef = useRef(true);
  const modeRef = useRef<Mode>("sim");
  const stepsRef = useRef(2);
  const sigmaRef = useRef(1.0);
  const busyRef = useRef(false);
  const lastFrameAtRef = useRef(performance.now());

  const [mode, setMode] = useState<Mode>("sim");
  const [running, setRunning] = useState(true);
  const [modelState, setModelState] = useState<ModelState>("loading");
  const [denoiseSteps, setDenoiseSteps] = useState(2);
  const [sigma, setSigma] = useState(1.0);
  const [fps, setFps] = useState(0);
  const [actionLabel, setActionLabel] = useState("noop");
  const [score, setScore] = useState(0);

  const modelStatus = useMemo(() => {
    if (modelState === "ready") return "model ready";
    if (modelState === "loading") return "loading model";
    if (modelState === "missing") return "model not published yet";
    return "model load failed";
  }, [modelState]);

  const syncCanvases = useCallback((frame: RgbFrame) => {
    if (canvasRef.current) drawFrame(canvasRef.current, frame);
    contextRef.current.forEach((thumb, index) => {
      const canvas = thumbRefs.current[index];
      if (canvas) drawFrame(canvas, thumb);
    });
  }, []);

  const reset = useCallback(() => {
    simRef.current.reset();
    contextRef.current = seedContext(simRef.current);
    setScore(simRef.current.score);
    syncCanvases(contextRef.current[contextRef.current.length - 1]);
  }, [syncCanvases]);

  useEffect(() => {
    modeRef.current = mode;
  }, [mode]);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    stepsRef.current = denoiseSteps;
  }, [denoiseSteps]);

  useEffect(() => {
    sigmaRef.current = sigma;
  }, [sigma]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (["ArrowLeft", "ArrowRight", "KeyA", "KeyD", "Space"].includes(event.code)) {
        event.preventDefault();
      }
      if (event.code === "Space") setRunning((value) => !value);
      keysRef.current.add(event.code);
    };
    const onKeyUp = (event: KeyboardEvent) => {
      keysRef.current.delete(event.code);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadModel() {
      try {
        const response = await fetch(MODEL_URL, { cache: "no-cache" });
        if (!response.ok) {
          if (!cancelled) setModelState("missing");
          return;
        }
        const buffer = await response.arrayBuffer();
        if (!looksLikeOnnx(buffer, response.headers.get("content-type"))) {
          if (!cancelled) setModelState("missing");
          return;
        }
        const sampler = await DiffusionSampler.create(new Uint8Array(buffer));
        if (!cancelled) {
          samplerRef.current = sampler;
          setModelState("ready");
        }
      } catch (error) {
        console.warn("Could not load neural model", error);
        if (!cancelled) setModelState("error");
      }
    }

    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let stopped = false;

    async function stepLoop() {
      if (stopped) return;
      if (!runningRef.current || busyRef.current) {
        window.setTimeout(stepLoop, 16);
        return;
      }

      busyRef.current = true;
      const action = currentAction(keysRef.current);
      setActionLabel(describeAction(action));

      try {
        let frame: RgbFrame;
        const sampler = samplerRef.current;
        if (modeRef.current === "neural" && sampler) {
          frame = await sampler.sample(contextRef.current, action, stepsRef.current, sigmaRef.current);
        } else {
          simRef.current.step(action);
          frame = simRef.current.renderRgb();
          setScore(simRef.current.score);
        }

        contextRef.current = pushContext(contextRef.current, frame);
        syncCanvases(frame);

        const now = performance.now();
        const delta = now - lastFrameAtRef.current;
        if (delta > 0) setFps(Math.round(1000 / delta));
        lastFrameAtRef.current = now;
      } finally {
        busyRef.current = false;
        window.setTimeout(stepLoop, modeRef.current === "neural" ? 0 : 24);
      }
    }

    syncCanvases(contextRef.current[contextRef.current.length - 1]);
    stepLoop();
    return () => {
      stopped = true;
    };
  }, [syncCanvases]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Tiny Real-Time World Model</h1>
          <p>Action-conditioned diffusion, rolled forward in your browser.</p>
        </div>
        <a
          className="github-link"
          href="https://github.com/silentvoice/tiny-real-time-world-model"
          target="_blank"
          rel="noreferrer"
          aria-label="Open GitHub repository"
        >
          <Github size={18} />
          GitHub
        </a>
      </header>

      <section className="play-layout">
        <div className="stage">
          <canvas ref={canvasRef} className="game-canvas" width={640} height={640} />
          <div className="context-strip" aria-label="Recent context frames">
            {Array.from({ length: CONTEXT_FRAMES }).map((_, index) => (
              <canvas
                key={index}
                ref={(element) => {
                  thumbRefs.current[index] = element;
                }}
                width={96}
                height={96}
              />
            ))}
          </div>
        </div>

        <aside className="control-panel" aria-label="World model controls">
          <div className="panel-section">
            <div className="section-title">World</div>
            <div className="segmented">
              <button
                type="button"
                className={mode === "sim" ? "active" : ""}
                onClick={() => setMode("sim")}
              >
                Real simulator
              </button>
              <button
                type="button"
                className={mode === "neural" ? "active" : ""}
                disabled={modelState !== "ready"}
                onClick={() => setMode("neural")}
              >
                Neural world
              </button>
            </div>
          </div>

          <div className="panel-section controls-row">
            <button
              type="button"
              className="icon-button"
              aria-label={running ? "Pause" : "Play"}
              onClick={() => setRunning((value) => !value)}
            >
              {running ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button type="button" className="icon-button" aria-label="Reset" onClick={reset}>
              <RotateCcw size={18} />
            </button>
            <div className="status-chip">
              <Sparkles size={15} />
              {modelStatus}
            </div>
          </div>

          <label className="slider-field">
            <span>Denoise steps</span>
            <strong>{denoiseSteps}</strong>
            <input
              type="range"
              min="1"
              max="6"
              step="1"
              value={denoiseSteps}
              onChange={(event) => setDenoiseSteps(Number(event.target.value))}
            />
          </label>

          <label className="slider-field">
            <span>Noise</span>
            <strong>{sigma.toFixed(2)}</strong>
            <input
              type="range"
              min="0.25"
              max="1.5"
              step="0.05"
              value={sigma}
              onChange={(event) => setSigma(Number(event.target.value))}
            />
          </label>

          <div className="metrics">
            <div>
              <span>FPS</span>
              <strong>{fps}</strong>
            </div>
            <div>
              <span>Action</span>
              <strong>{actionLabel}</strong>
            </div>
            <div>
              <span>Score</span>
              <strong>{mode === "sim" ? score : "dream"}</strong>
            </div>
          </div>

          <div className="keyboard">
            <kbd>A</kbd>
            <kbd>←</kbd>
            <span>move left</span>
            <kbd>D</kbd>
            <kbd>→</kbd>
            <span>move right</span>
            <kbd>Space</kbd>
            <span>pause</span>
          </div>
        </aside>
      </section>
    </main>
  );
}
