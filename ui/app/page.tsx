"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { NetworkField } from "./network-field";
import {
  EMPTY_METRICS,
  type OrganismPayload,
} from "./organism-types";

const EXAMPLES = [
  "to be, or not to be",
  "the field remembers",
  "what happens during silence?",
];

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
  const [organism, setOrganism] = useState<OrganismPayload | null>(null);
  const [running, setRunning] = useState(true);
  const [pending, setPending] = useState(false);
  const [selectedCell, setSelectedCell] = useState(17);
  const [fieldMode, setFieldMode] = useState<
    "connecting" | "offline" | "live-checkpoint"
  >("connecting");
  const [notice, setNotice] = useState(
    "Connecting to the local checkpoint bridge…",
  );
  const clockBusy = useRef(false);

  const metrics = organism?.metrics ?? EMPTY_METRICS;
  const topology = organism?.topology ?? null;
  const cellCount = organism?.checkpoint.cells ?? 0;
  const dendrites = organism?.checkpoint.dendrites ?? 0;
  const tick = organism?.clock.ticks ?? 0;
  const activeCharacter = organism?.clock.lastOutput ?? "";

  const sensoryCells = useMemo(
    () => new Set(topology?.sensoryCells ?? []),
    [topology?.sensoryCells],
  );
  const outputCells = useMemo(
    () => new Set(topology?.outputCells ?? []),
    [topology?.outputCells],
  );
  const selectedRole = sensoryCells.has(selectedCell)
    ? "sensory"
    : outputCells.has(selectedCell)
      ? "output"
      : "interior";
  const selectedActivity = topology?.cellActivity[selectedCell] ?? 0;
  const selectedEnergy = topology?.cellEnergy[selectedCell] ?? 0;

  const applyPayload = useCallback(
    (payload: OrganismPayload, appendOutput: boolean) => {
      setOrganism(payload);
      setFieldMode("live-checkpoint");
      setSelectedCell((current) =>
        payload.checkpoint.cells <= current ? 0 : current,
      );
      if (appendOutput && payload.output) {
        setOutput((current) =>
          (current + payload.output).slice(-2400),
        );
      }
    },
    [],
  );

  useEffect(() => {
    let cancelled = false;
    async function connect() {
      try {
        const response = await fetch("/api/organism", {
          cache: "no-store",
        });
        if (!response.ok) throw new Error("bridge offline");
        const payload = (await response.json()) as OrganismPayload;
        if (cancelled) return;
        applyPayload(payload, false);
        setNotice(
          payload.checkpoint.metabolismEnabled
            ? "Live connectome loaded. The organism clock is running through genuine no-input intervals."
            : "Live connectome loaded. This language checkpoint has metabolism disabled, so silent ticks do not deplete energy.",
        );
      } catch {
        if (cancelled) return;
        setFieldMode("offline");
        setNotice(
          "Checkpoint bridge offline. Start `python -m sol.serve`; no synthetic network is being shown.",
        );
      }
    }
    void connect();
    return () => {
      cancelled = true;
    };
  }, [applyPayload]);

  useEffect(() => {
    if (!running || pending || fieldMode !== "live-checkpoint") return;
    let cancelled = false;

    async function advance() {
      if (clockBusy.current) return;
      clockBusy.current = true;
      try {
        const response = await fetch("/api/organism", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ steps: 1, temperature: 0.8 }),
        });
        if (!response.ok) throw new Error("clock failed");
        const payload = (await response.json()) as OrganismPayload;
        if (cancelled) return;
        applyPayload(payload, true);
        setNotice(
          payload.checkpoint.metabolismEnabled
            ? "Clock running · no sensory token · output sampled from the evolving field."
            : "Clock running · no sensory token · metabolism is disabled in this checkpoint.",
        );
      } catch {
        if (!cancelled) {
          setFieldMode("offline");
          setNotice("The local organism clock lost its checkpoint bridge.");
        }
      } finally {
        clockBusy.current = false;
      }
    }

    void advance();
    const interval = window.setInterval(() => void advance(), 250);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [applyPayload, fieldMode, pending, running]);

  async function generate() {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || pending) return;
    setPending(true);
    setOutput("");
    setNotice("Streaming the prompt through the sensory organ…");

    try {
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: cleanPrompt, length: 180 }),
      });
      if (!response.ok) throw new Error("generation request failed");
      const result = (await response.json()) as OrganismPayload;
      if (result.mode !== "live-checkpoint") {
        throw new Error("checkpoint bridge unavailable");
      }
      applyPayload(result, false);
      setNotice("Reading the real output organ one character at a time.");
      for (let index = 0; index < (result.output?.length ?? 0); index += 1) {
        setOutput(result.output?.slice(0, index + 1) ?? "");
        await new Promise((resolve) => window.setTimeout(resolve, 18));
      }
      setNotice(
        `Stimulus complete · ${result.checkpoint.name} at training update ${result.checkpoint.updates.toLocaleString()} · autonomous clock resumes.`,
      );
    } catch {
      setNotice(
        "The live field could not answer. Check the local Python bridge.",
      );
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
          {fieldMode === "live-checkpoint"
            ? "CHECKPOINT ONLINE"
            : fieldMode === "connecting"
              ? "CONNECTING"
              : "BRIDGE OFFLINE"}
          <span className="tick-count">
            ORGANISM TICK {tick.toLocaleString()}
          </span>
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">Actual connectome · persistent clock</p>
          <h1>Watch the organism run.</h1>
        </div>
        <p className="intro-copy">
          Every visible node and dendrite belongs to the loaded checkpoint.
          Edge brightness is measured message flow from its latest tick; the
          geometric layout is only a readable projection.
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
              <span>{cellCount || "—"} cells</span>
              <span>
                {cellCount && dendrites ? cellCount * dendrites : "—"} dendrites
              </span>
              <button
                type="button"
                className="text-button"
                onClick={() => {
                  setRunning((value) => !value);
                  setNotice(
                    running
                      ? "Organism frozen. Persistent state is retained exactly."
                      : "Organism clock resumed.",
                  );
                }}
                aria-pressed={!running}
              >
                {running ? "Freeze organism" : "Run organism"}
              </button>
            </div>
          </div>
          <NetworkField
            topology={topology}
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
            <span>
              <i className="positive-edge" /> Positive edge
            </span>
            <span>
              <i className="negative-edge" /> Negative edge
            </span>
            <span className="active-token">
              LAST OUTPUT <b>{JSON.stringify(activeCharacter || "—")}</b>
            </span>
          </div>
        </article>

        <aside className="telemetry-panel" aria-label="Organism telemetry">
          <div className="panel-heading compact">
            <div>
              <p className="panel-kicker">ORGANISM STATE</p>
              <h2>Telemetry</h2>
            </div>
            <span className="sampling-badge">
              {running ? "4 Hz clock" : "frozen"}
            </span>
          </div>

          <div
            className="energy-gauge"
            style={
              {
                "--energy-angle": `${Math.max(0, Math.min(1, metrics.energy)) * 360}deg`,
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
              detail="zero during silence"
            />
            <Metric
              label="PERPLEXITY"
              value={
                Number.isFinite(metrics.perplexity)
                  ? metrics.perplexity.toFixed(2)
                  : "—"
              }
              detail="last stimulus"
            />
            <Metric
              label="CELL CREDIT"
              value={metrics.cellCredit.toExponential(1)}
              detail="last prompt gradient"
            />
            <Metric
              label="EDGE FLOW"
              value={metrics.edgeFlow.toExponential(1)}
              detail="latest real tick"
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
                <dt>ACTIVITY</dt>
                <dd>{selectedActivity.toFixed(3)}</dd>
              </div>
              <div>
                <dt>ENERGY</dt>
                <dd>{selectedEnergy.toFixed(3)}</dd>
              </div>
              <div>
                <dt>DENDRITES</dt>
                <dd>{dendrites || "—"}</dd>
              </div>
              <div>
                <dt>VIABILITY</dt>
                <dd>
                  {(topology?.cellViability[selectedCell] ?? 0).toFixed(3)}
                </dd>
              </div>
              <div>
                <dt>METABOLISM</dt>
                <dd>
                  {organism?.checkpoint.metabolismEnabled ? "enabled" : "disabled"}
                </dd>
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
            <span>⌘ + ENTER TO STIMULATE</span>
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
              disabled={
                !prompt.trim() ||
                pending ||
                fieldMode !== "live-checkpoint"
              }
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
            <span
              className={
                running || pending ? "streaming active" : "streaming"
              }
            >
              {pending ? "STIMULATED" : running ? "AUTONOMOUS" : "FROZEN"}
            </span>
          </div>
          <div
            className="response-text"
            aria-live="polite"
            data-testid="response-output"
          >
            {output || (
              <span>
                Waiting for the first no-input clock tick from the output organ.
              </span>
            )}
            {(running || pending) && (
              <i className="cursor" aria-hidden="true" />
            )}
          </div>
          <p className="notice">{notice}</p>
        </div>
      </section>

      <footer>
        <p>
          EXACT TOPOLOGY · MEASURED FLOW · NO-INPUT CLOCK · PERSISTENT MEMORY
        </p>
        <span>Prototype field 0.2</span>
      </footer>
    </main>
  );
}
