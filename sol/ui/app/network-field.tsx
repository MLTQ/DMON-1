"use client";

import { useEffect, useMemo, useRef } from "react";

import type { OrganismTopology } from "./organism-types";

type Role = "sensory" | "interior" | "output";

type NodeSpec = {
  id: number;
  role: Role;
  x: number;
  y: number;
};

type EdgeSpec = {
  source: number;
  target: number;
  slot: number;
  weight: number;
  flow: number;
};

function makeNodes(topology: OrganismTopology): NodeSpec[] {
  const sensory = new Set(topology.sensoryCells);
  const output = new Set(topology.outputCells);
  const interior = topology.sources
    .map((_, id) => id)
    .filter((id) => !sensory.has(id) && !output.has(id));
  const interiorIndex = new Map(interior.map((id, index) => [id, index]));
  const columns = Math.max(1, Math.ceil(Math.sqrt(interior.length * 1.45)));
  const rows = Math.max(1, Math.ceil(interior.length / columns));

  return topology.sources.map((_, id) => {
    if (sensory.has(id)) {
      const index = topology.sensoryCells.indexOf(id);
      return {
        id,
        role: "sensory",
        x: 0.055,
        y: (index + 1) / (topology.sensoryCells.length + 1),
      };
    }
    if (output.has(id)) {
      const index = topology.outputCells.indexOf(id);
      return {
        id,
        role: "output",
        x: 0.945,
        y: (index + 1) / (topology.outputCells.length + 1),
      };
    }
    const index = interiorIndex.get(id) ?? 0;
    const column = index % columns;
    const row = Math.floor(index / columns);
    return {
      id,
      role: "interior",
      x: columns === 1 ? 0.5 : 0.17 + (column / (columns - 1)) * 0.66,
      y: rows === 1 ? 0.5 : 0.1 + (row / (rows - 1)) * 0.8,
    };
  });
}

function makeEdges(topology: OrganismTopology): EdgeSpec[] {
  return topology.sources.flatMap((sources, target) =>
    sources.map((source, slot) => ({
      source,
      target,
      slot,
      weight: topology.weights[target]?.[slot] ?? 0,
      flow: topology.edgeFlow[target]?.[slot] ?? 0,
    })),
  );
}

export function NetworkField({
  topology,
  running,
  selectedCell,
  onSelectCell,
}: {
  topology: OrganismTopology | null;
  running: boolean;
  selectedCell: number;
  onSelectCell: (cell: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const nodes = useMemo(() => (topology ? makeNodes(topology) : []), [topology]);
  const edges = useMemo(() => (topology ? makeEdges(topology) : []), [topology]);
  const stateRef = useRef({ topology, running, selectedCell, nodes, edges });

  useEffect(() => {
    stateRef.current = { topology, running, selectedCell, nodes, edges };
  }, [topology, running, selectedCell, nodes, edges]);

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

    const draw = () => {
      const rect = canvas.getBoundingClientRect();
      const width = rect.width;
      const height = rect.height;
      const current = stateRef.current;

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

      if (!current.topology) {
        context.fillStyle = "#71897d";
        context.font = "11px monospace";
        context.textAlign = "center";
        context.fillText(
          "CHECKPOINT BRIDGE OFFLINE · NO TOPOLOGY TO DISPLAY",
          width / 2,
          height / 2,
        );
        animationRef.current = requestAnimationFrame(draw);
        return;
      }

      const maxFlow = Math.max(
        0.000001,
        ...current.edges.map((edge) => edge.flow),
      );
      current.edges.forEach((edge) => {
        const source = current.nodes[edge.source];
        const target = current.nodes[edge.target];
        if (!source || !target) return;
        const sx = source.x * width;
        const sy = source.y * height;
        const tx = target.x * width;
        const ty = target.y * height;
        const normalizedFlow = Math.min(1, edge.flow / maxFlow);
        const selected =
          edge.source === current.selectedCell ||
          edge.target === current.selectedCell;
        const alpha = 0.035 + normalizedFlow * 0.48 + (selected ? 0.16 : 0);

        context.beginPath();
        if (edge.source === edge.target) {
          context.arc(sx + 7, sy - 7, 7, 0.25 * Math.PI, 2.1 * Math.PI);
        } else {
          context.moveTo(sx, sy);
          context.lineTo(tx, ty);
        }
        context.strokeStyle =
          edge.weight >= 0
            ? `rgba(112, 223, 226, ${alpha})`
            : `rgba(239, 188, 125, ${alpha})`;
        context.lineWidth = 0.45 + normalizedFlow * 2.2 + (selected ? 0.4 : 0);
        context.stroke();
      });

      current.nodes.forEach((node) => {
        const x = node.x * width;
        const y = node.y * height;
        const activity = Math.min(
          1,
          current.topology?.cellActivity[node.id] ?? 0,
        );
        const energy = Math.min(
          1,
          current.topology?.cellEnergy[node.id] ?? 0,
        );
        const selected = node.id === current.selectedCell;
        const radius = selected ? 7.2 : 4.2 + activity * 2.4;

        if (activity > 0.01 || selected) {
          const glow = context.createRadialGradient(
            x,
            y,
            0,
            x,
            y,
            selected ? 21 : 12 + activity * 8,
          );
          glow.addColorStop(
            0,
            selected
              ? "rgba(217,255,154,.32)"
              : `rgba(102,237,180,${0.08 + activity * 0.24})`,
          );
          glow.addColorStop(1, "rgba(79,211,171,0)");
          context.fillStyle = glow;
          context.beginPath();
          context.arc(x, y, selected ? 21 : 12 + activity * 8, 0, Math.PI * 2);
          context.fill();
        }

        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        if (node.role === "sensory") {
          context.fillStyle = `rgba(112,223,226,${0.35 + activity * 0.65})`;
        } else if (node.role === "output") {
          context.fillStyle = `rgba(217,255,154,${0.35 + activity * 0.65})`;
        } else {
          context.fillStyle = `rgba(132,237,177,${0.24 + activity * 0.76})`;
        }
        context.fill();
        context.strokeStyle = selected
          ? "#f2ffd3"
          : `rgba(222,255,230,${0.12 + energy * 0.45})`;
        context.lineWidth = selected ? 1.6 : 0.8;
        context.stroke();
      });

      context.fillStyle = "rgba(113,137,125,.72)";
      context.font = "9px monospace";
      context.textAlign = "left";
      context.fillText(
        current.running
          ? "CLOCK RUNNING · FLOW FROM LATEST REAL TICK"
          : "CLOCK FROZEN · LAST MEASURED FLOW RETAINED",
        14,
        height - 14,
      );
      context.textAlign = "right";
      context.fillText("ABSTRACT LAYOUT · EXACT CONNECTIVITY", width - 14, height - 14);
      animationRef.current = requestAnimationFrame(draw);
    };

    animationRef.current = requestAnimationFrame(draw);
    return () => {
      observer.disconnect();
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, []);

  function selectNode(event: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas || nodes.length === 0) return;
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
    if (nearest >= 0 && distance < 0.045) onSelectCell(nearest);
  }

  return (
    <canvas
      ref={canvasRef}
      className="network-canvas"
      aria-label="Actual checkpoint connectome. Edge brightness is measured message flow from the latest organism tick."
      onPointerDown={selectNode}
      data-testid="network-canvas"
    />
  );
}
