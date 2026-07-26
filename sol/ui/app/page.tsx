"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Metrics = {
  energy: number;
  novelty: number;
  cellCredit: number;
  edgeCredit: number;
  perplexity: number;
};

type NodeSpec = {
  id: number;
  role: "sensory" | "interior" | "output";
  x: number;
  y: number;
};

const CELL_COUNT = 36;
const DENDRITES = 4;
const EXAMPLES = [
  "to be, or not to be",
  "the field remembers",
  "what happens during silence?",
];

const INITIAL_METRICS: Metrics = {
  energy: 0.62,
  novelty: 0.18,
  cellCredit: 0.000158,
  edgeCredit: 0.000114,
  perplexity: 1.94,
};

function makeNodes(): NodeSpec[] {
  return Array.from({ length: CELL_COUNT }, (_, id) => {
    if (id < 4) {
      return { id, role: "sensory", x: 0.075, y: 0.2 + id * 0.2 };
    }
    if (id >= CELL_COUNT - 4) {
      return {
        id,
        role: "output",
        x: 0.925,
        y: 0.2 + (id - CELL_COUNT + 4) * 0.2,
      };
    }
    const local = id - 4;
    const column = local % 7;
    const row = Math.floor(local / 7);
    return {
      id,
      role: "interior",
      x: 0.19 + column * 0.105 + (row % 2) * 0.018,
      y: 0.17 + row * 0.22 + Math.sin(id * 2.17) * 0.025,
    };
  });
}

function makeEdges() {
  const edges: Array<{ source: number; target: number; slot: number }> = [];
  for (let target = 0; target < CELL_COUNT; target += 1) {
    const sources = [
      target,
      (target - 1 + CELL_COUNT) % CELL_COUNT,
      (target - 6 + CELL_COUNT) % CELL_COUNT,
      (target * 7 + 3) % CELL_COUNT,
    ];
    if (target >= CELL_COUNT - 4) {
      sources[3] = target - (CELL_COUNT - 4);
    }
    sources.forEach((source, slot) => edges.push({ source, target, slot }));
  }
  return edges;
}

function NetworkField({
  activeCharacter,
  tick,
  running,
  selectedCell,
  onSelectCell,
}: {
  activeCharacter: string;
  tick: number;
  running: boolean;
  selectedCell: number;
  onSelectCell: (cell: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodes = useMemo(() => makeNodes(), []);
  const edges = useMemo(() => makeEdges(), []);
  const animationRef = useRef<number | null>(null);
  const stateRef = useRef({ activeCharacter, tick, running, selectedCell });

  useEffect(() => {
    stateRef.current = { activeCharacter, tick, running, selectedCell };
  }, [activeCharacter, tick, running, selectedCell]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      context.setTransform(scale, 0, 0, scale, 0, 0);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const draw = (time: number) => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const current = stateRef.current;
      const code = current.activeCharacter.charCodeAt(0) || 7;

      context.clearRect(0, 0, width, height);
      context.fillStyle = "#07100f";
      context.fillRect(0, 0, width, height);

      context.strokeStyle = "rgba(172, 237, 204, 0.035)";
      context.lineWidth = 1;
      for (let x = 24; x < width; x += 32) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let y = 24; y < height; y += 32) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      edges.forEach((edge, edgeIndex) => {
        const source = nodes[edge.source];
        const target = nodes[edge.target];
        const sx = source.x * width;
        const sy = source.y * height;
        const tx = target.x * width;
        const ty = target.y * height;
        const active =
          ((edgeIndex * 17 + code + current.tick * 3) % 19) < 5;

        context.beginPath();
        context.moveTo(sx, sy);
        context.lineTo(tx, ty);
        context.strokeStyle = active
          ? "rgba(117, 232, 178, 0.29)"
          : "rgba(126, 159, 147, 0.095)";
        context.lineWidth = active ? 1.15 : 0.65;
        context.stroke();

        if (current.running && active && edge.source !== edge.target) {
          const phase =
            (time * 0.00021 + ((edgeIndex + code) % 13) / 13) % 1;
          const px = sx + (tx - sx) * phase;
          const py = sy + (ty - sy) * phase;
          context.beginPath();
          context.arc(px, py, 1.8, 0, Math.PI * 2);
          context.fillStyle =
            edge.slot % 2 === 0 ? "#8bf5bd" : "#58d7da";
          context.fill();
        }
      });

      nodes.forEach((node) => {
        const x = node.x * width;
        const y = node.y * height;
        const oscillation =
          0.5 + 0.5 * Math.sin(time * 0.002 + node.id * 1.91 + code);
        const selected = node.id === current.selectedCell;
        const stimulated =
          current.running &&
          ((node.id * 11 + code + current.tick) % 17) < 6;
        const radius = selected ? 8.5 : stimulated ? 6.4 : 4.6;

        if (selected || stimulated) {
          const glow = context.createRadialGradient(
            x,
            y,
            0,
            x,
            y,
            selected ? 24 : 17,
          );
          glow.addColorStop(
            0,
            selected
              ? "rgba(217,255,154,.34)"
              : "rgba(102,237,180,.20)",
          );
          glow.addColorStop(1, "rgba(79,211,171,0)");
          context.fillStyle = glow;
          context.beginPath();
          context.arc(x, y, selected ? 24 : 17, 0, Math.PI * 2);
          context.fill();
        }

        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        if (node.role === "sensory") {
          context.fillStyle = stimulated ? "#72e5e8" : "#285c60";
        } else if (node.role === "output") {
          context.fillStyle = stimulated ? "#d9ff9a" : "#687a45";
        } else {
          const green = Math.round(105 + oscillation * 86);
          context.fillStyle = stimulated
            ? `rgb(92, ${green + 44}, 145)`
            : `rgb(36, ${green}, 86)`;
        }
        context.fill();
        context.strokeStyle = selected
          ? "#f2ffd3"
          : "rgba(222,255,230,.28)";
        context.lineWidth = selected ? 1.5 : 0.7;
        context.stroke();
      });

      animationRef.current = requestAnimationFrame(draw);
    };

    animationRef.current = requestAnimationFrame(draw);
    return () => {
      observer.disconnect();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [edges, nodes]);

  function selectNode(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    let nearest = -1;
    let distance = Number.POSITIVE_INFINITY;
    nodes.forEach((node) => {
      const value = Math.hypot(node.x - x, node.y - y);
      if (value < distance) {
        nearest = node.id;
        distance = value;
      }
    });
    if (nearest >= 0 && distance < 0.04) onSelectCell(nearest);
  }

  return (
    <canvas
      ref={canvasRef}
      className="network-canvas"
      aria-label="Animated directed neural field. Click a cell to inspect it."
      onPointerDown={selectNode}
      data-testid="network-canvas"
    />
  );
}

function Metric({
  label,
  value,
  detail,
}: {
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export default function Home() {
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const [output, setOutput] = useState("");
  const [activeCharacter, setActiveCharacter] = useState("t");
  const [tick, setTick] = useState(21_408);
  const [running, setRunning] = useState(true);
  const [pending, setPending] = useState(false);
  const [selectedCell, setSelectedCell] = useState(17);
  const [metrics, setMetrics] = useState<Metrics>(INITIAL_METRICS);
  const [notice, setNotice] = useState(
    "Ready. This hosted console runs a deterministic browser demonstration.",
  );

  const selectedRole =
    selectedCell < 4
      ? "sensory"
      : selectedCell >= CELL_COUNT - 4
        ? "output"
        : "interior";

  async function generate() {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || pending) return;
    setPending(true);
    setRunning(true);
    setOutput("");
    setNotice("Streaming the prompt into the field…");

    try {
      for (const character of cleanPrompt.slice(-16)) {
        setActiveCharacter(character);
        setTick((value) => value + 1);
        await new Promise((resolve) => window.setTimeout(resolve, 20));
      }

      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: cleanPrompt, length: 180 }),
      });
      if (!response.ok) throw new Error("generation request failed");
      const result = (await response.json()) as {
        output: string;
        metrics: Metrics;
        mode: string;
      };

      setNotice("Reading the output organ one character at a time.");
      for (let index = 0; index < result.output.length; index += 1) {
        const character = result.output[index];
        setOutput(result.output.slice(0, index + 1));
        setActiveCharacter(character);
        setTick((value) => value + 1);
        if (index % 4 === 0) {
          setMetrics((current) => ({
            ...current,
            energy: Math.max(
              0.12,
              result.metrics.energy +
                Math.sin(index * 0.41) * 0.035,
            ),
            novelty: Math.max(
              0.01,
              result.metrics.novelty +
                Math.cos(index * 0.29) * 0.018,
            ),
          }));
        }
        await new Promise((resolve) => window.setTimeout(resolve, 18));
      }
      setMetrics(result.metrics);
      setNotice(
        result.mode === "browser-demo"
          ? "Complete · browser demonstration. The local PyTorch bridge can replace this endpoint."
          : "Complete · response generated by the live SOL organism.",
      );
    } catch {
      setNotice("The field could not answer. Try the prompt again.");
    } finally {
      setPending(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            S
          </span>
          <div>
            <p>SOL / CHARACTER ORGAN</p>
            <span>Sparse Organism Laboratory</span>
          </div>
        </div>
        <div className="system-status">
          <span className="status-light" aria-hidden="true" />
          DEMO FIELD ONLINE
          <span className="tick-count">TICK {tick.toLocaleString()}</span>
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">Persistent neural field · streamed input</p>
          <h1>Watch a thought move.</h1>
        </div>
        <p className="intro-copy">
          Characters enter through sensory cells. Directed axons carry activity
          across a persistent field. The output organ answers one character at
          a time.
        </p>
      </section>

      <section className="instrument-grid">
        <article className="field-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">LIVE CONNECTOME</p>
              <h2>Directed field</h2>
            </div>
            <div className="field-actions">
              <span>{CELL_COUNT} cells</span>
              <span>{CELL_COUNT * DENDRITES} dendrites</span>
              <button
                type="button"
                className="text-button"
                onClick={() => setRunning((value) => !value)}
                aria-pressed={!running}
              >
                {running ? "Pause field" : "Resume field"}
              </button>
            </div>
          </div>
          <NetworkField
            activeCharacter={activeCharacter}
            tick={tick}
            running={running}
            selectedCell={selectedCell}
            onSelectCell={setSelectedCell}
          />
          <div className="field-legend">
            <span>
              <i className="sensory-dot" /> Sensory
            </span>
            <span>
              <i className="interior-dot" /> Interior
            </span>
            <span>
              <i className="output-dot" /> Output
            </span>
            <span className="active-token">
              ACTIVE TOKEN <b>{JSON.stringify(activeCharacter)}</b>
            </span>
          </div>
        </article>

        <aside className="telemetry-panel" aria-label="Organism telemetry">
          <div className="panel-heading compact">
            <div>
              <p className="panel-kicker">ORGANISM STATE</p>
              <h2>Telemetry</h2>
            </div>
            <span className="sampling-badge">60 Hz</span>
          </div>

          <div
            className="energy-gauge"
            style={
              {
                "--energy-angle": `${metrics.energy * 360}deg`,
              } as React.CSSProperties
            }
          >
            <div>
              <strong>{Math.round(metrics.energy * 100)}</strong>
              <span>% ENERGY</span>
            </div>
          </div>

          <div className="metrics-grid">
            <Metric
              label="NOVELTY"
              value={metrics.novelty.toFixed(3)}
              detail="input delta"
            />
            <Metric
              label="PERPLEXITY"
              value={metrics.perplexity.toFixed(2)}
              detail="character stream"
            />
            <Metric
              label="CELL CREDIT"
              value={metrics.cellCredit.toExponential(1)}
              detail="backward signal"
            />
            <Metric
              label="EDGE CREDIT"
              value={metrics.edgeCredit.toExponential(1)}
              detail="signed synapse"
            />
          </div>

          <div className="cell-inspector">
            <div>
              <span>SELECTED CELL</span>
              <strong>#{selectedCell.toString().padStart(2, "0")}</strong>
            </div>
            <dl>
              <div>
                <dt>ROLE</dt>
                <dd>{selectedRole}</dd>
              </div>
              <div>
                <dt>STATE</dt>
                <dd>{running ? "firing" : "retained"}</dd>
              </div>
              <div>
                <dt>DENDRITES</dt>
                <dd>{DENDRITES}</dd>
              </div>
            </dl>
          </div>
        </aside>
      </section>

      <section className="interaction-panel">
        <div className="composer">
          <div className="composer-heading">
            <div>
              <p className="panel-kicker">STIMULUS</p>
              <h2>Talk to the field</h2>
            </div>
            <span>⌘ + ENTER TO RUN</span>
          </div>
          <label htmlFor="prompt">Prompt</label>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                void generate();
              }
            }}
            placeholder="Give the organism a character stream…"
            maxLength={500}
          />
          <div className="prompt-options">
            <div className="example-prompts" aria-label="Example prompts">
              {EXAMPLES.map((example) => (
                <button
                  type="button"
                  key={example}
                  onClick={() => setPrompt(example)}
                >
                  {example}
                </button>
              ))}
            </div>
            <button
              type="button"
              className="run-button"
              onClick={() => void generate()}
              disabled={!prompt.trim() || pending}
            >
              {pending ? "FIELD RUNNING" : "STIMULATE FIELD"}
              <span aria-hidden="true">→</span>
            </button>
          </div>
        </div>

        <div className="response">
          <div className="response-heading">
            <div>
              <p className="panel-kicker">OUTPUT ORGAN</p>
              <h2>Response stream</h2>
            </div>
            <span className={pending ? "streaming active" : "streaming"}>
              {pending ? "STREAMING" : "READY"}
            </span>
          </div>
          <div
            className="response-text"
            aria-live="polite"
            data-testid="response-output"
          >
            {output || (
              <span>
                The field is retaining state. Send a prompt to stimulate it.
              </span>
            )}
            {pending && <i className="cursor" aria-hidden="true" />}
          </div>
          <p className="notice">{notice}</p>
        </div>
      </section>

      <footer>
        <p>
          ONE SHARED STATE · DIRECTED TRAFFIC · CONTINUOUS CREDIT · PERSISTENT
          MEMORY
        </p>
        <span>Prototype field 0.1</span>
      </footer>
    </main>
  );
}
