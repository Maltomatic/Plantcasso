"use client";

/**
 * Dashboard.tsx — client component.
 *
 * Connects to the server-side serial stream (`/api/stream`) via EventSource,
 * keeps a rolling window of the last MAX_POINTS readings, and renders the four
 * Recharts panels migrated from the original Flask `dashboard-2` (Plotly):
 *   1. Mean ± Std band
 *   2. Spike count
 *   3. Hjorth complexity
 *   4. Cluster (0 = stable, 1 = slight agitation, 2 = high agitation)
 */

import { useEffect, useRef, useState } from "react";
import {
  Area,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  LineChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

const MAX_POINTS = 100;

interface Reading {
  timestamp: number;
  mean: number;
  std: number;
  spike_count: number;
  hjorth: number;
  cluster: number;
  volts: number;
}

interface Point extends Reading {
  t: string; // HH:MM:SS label for the x-axis
  band: [number, number]; // [mean - std, mean + std]
}

// Functional state palette — these colors *are* the data. Stable green,
// amber unease, coral alarm. Matched to globals.css tokens.
const CLUSTER_COLORS = ["#3f9b57", "#c2872b", "#cf4b38"];
const CLUSTER_LABELS = ["Stable", "Slight Agitation", "High Agitation"];

// Shared chart theme so every instrument reads as one panel.
const GRID = "rgba(36,56,44,0.10)";
const AXIS = { fill: "#93937c", fontSize: 10, fontFamily: "var(--font-mono)" };
const CURSOR = { stroke: "rgba(36,56,44,0.22)" };
// Per-channel trace accents, tuned for contrast on parchment.
const C_POTENTIAL = "#3e7ca8";
const C_SPIKE = "#c2872b";
const C_HJORTH = "#3f9b57";

function fmtTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour12: false,
  });
}

// Custom tooltip — mono, instrument-styled, color-keyed to the series.
function ChartTip({
  active,
  payload,
  label,
  unit = "",
}: {
  active?: boolean;
  payload?: Array<{ name?: string; value?: number; color?: string }>;
  label?: string;
  unit?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-[var(--line-strong)] bg-[var(--surface-2)] px-3 py-2 font-mono text-[11px] shadow-xl">
      <div className="mb-1 text-[var(--sage)]">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-[var(--ink)]">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: p.color }}
          />
          <span className="text-[var(--sage)]">{p.name}</span>
          <span className="ml-auto tabular-nums">
            {typeof p.value === "number" ? p.value.toFixed(3) : p.value}
            {unit}
          </span>
        </div>
      ))}
    </div>
  );
}

function Panel({
  channel,
  title,
  hint,
  accent,
  children,
}: {
  channel: string;
  title: string;
  hint: string;
  accent: string;
  children: React.ReactNode;
}) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--surface)]/80 p-5 backdrop-blur-sm transition-colors hover:border-[var(--line-strong)]">
      <div
        className="pointer-events-none absolute inset-x-0 top-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${accent}66, transparent)` }}
      />
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <div>
          <div className="eyebrow" style={{ color: accent }}>
            {channel}
          </div>
          <h2 className="mt-1 font-display text-lg font-medium text-[var(--ink)]">
            {title}
          </h2>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--sage-dim)]">
          {hint}
        </span>
      </div>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [entered, setEntered] = useState(false);
  const [fading, setFading] = useState(false);
  const [points, setPoints] = useState<Point[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const es = new EventSource("/api/stream");
    esRef.current = es;

    es.onmessage = (ev) => {
      let r: Reading;
      try {
        r = JSON.parse(ev.data) as Reading;
      } catch {
        return;
      }
      const point: Point = {
        ...r,
        t: fmtTime(r.timestamp),
        band: [r.mean - r.std, r.mean + r.std],
      };
      setPoints((prev) => {
        const next = [...prev, point];
        return next.length > MAX_POINTS
          ? next.slice(next.length - MAX_POINTS)
          : next;
      });
    };

    es.onerror = () => {
      // EventSource auto-reconnects; nothing to do here.
    };

    return () => es.close();
  }, []);

  function enterDashboard() {
    setFading(true);
    setTimeout(() => setEntered(true), 1000);
  }

  // Live derived state for the hero vitals — the plant's current condition.
  const latest = points.length ? points[points.length - 1] : null;
  const connected = points.length > 0;
  const stateIdx = latest ? Math.max(0, Math.min(2, latest.cluster)) : 0;
  const stateColor = CLUSTER_COLORS[stateIdx];
  const stateLabel = CLUSTER_LABELS[stateIdx];

  if (!entered) {
    return (
      <div
        className="fixed inset-0 z-[1000] flex items-center justify-center overflow-hidden bg-[var(--bg)] transition-opacity duration-1000"
        style={{ opacity: fading ? 0 : 1 }}
      >
        <video
          autoPlay
          muted
          loop
          playsInline
          className="absolute h-full w-full object-cover opacity-40 mix-blend-multiply saturate-[0.75]"
        >
          <source src="/vid.mp4" type="video/mp4" />
        </video>
        {/* Parchment veil: keeps the botanical footage airy and the type legible */}
        <div className="absolute inset-0 bg-[var(--bg)]/35" />
        <div className="absolute inset-0 bg-[radial-gradient(120%_85%_at_50%_25%,transparent_25%,var(--bg))]" />

        <div className="relative z-10 mx-auto max-w-2xl px-6 text-center">
          <h1 className="font-display text-6xl font-light leading-[0.95] tracking-tight text-[var(--ink)] sm:text-7xl">
            Plant<span className="italic text-[var(--stable)]">casso</span>
          </h1>
          <p className="mx-auto mt-6 max-w-md font-mono text-sm leading-relaxed text-[var(--sage)]">
            A houseplant, wired and listening. Bioelectric signals clustered in
            real time into calm, unease, and alarm.
          </p>
          <button
            onClick={enterDashboard}
            className="group mt-10 inline-flex items-center gap-3 rounded-full border border-[var(--line-strong)] bg-[var(--surface)]/60 px-6 py-3 font-mono text-sm uppercase tracking-[0.18em] text-[var(--ink)] backdrop-blur-sm transition-all hover:border-[var(--stable)] hover:bg-[var(--surface-2)]"
          >
            Begin monitoring
            <span className="transition-transform group-hover:translate-x-1">→</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-[1400px] px-5 pb-16 pt-6 sm:px-8">
      {/* Header — wordmark + live telemetry status */}
      <header className="flex flex-wrap items-center justify-between gap-4 border-b border-[var(--line)] pb-5">
        <div className="flex items-baseline gap-3">
          <span className="font-display text-2xl font-light tracking-tight text-[var(--ink)]">
            Plant<span className="italic text-[var(--stable)]">casso</span>
          </span>
          <span className="hidden font-mono text-[11px] uppercase tracking-[0.2em] text-[var(--sage-dim)] sm:inline">
            Biosignal Telemetry
          </span>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--surface)]/60 px-3 py-1.5">
          <span
            className={`inline-block h-2 w-2 rounded-full ${connected ? "vital-dot" : ""}`}
            style={{ background: connected ? "var(--stable)" : "var(--sage-dim)" }}
          />
          <span className="font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--sage)]">
            {connected ? "Live · Serial" : "Awaiting signal"}
          </span>
        </div>
      </header>

      {/* Hero vitals — the signature: a breathing patient-monitor readout */}
      <section
        className="relative mt-6 overflow-hidden rounded-3xl border bg-[var(--surface)]/70 p-6 sm:p-8"
        style={{ borderColor: `${stateColor}40` }}
      >
        <div
          className="pointer-events-none absolute inset-0"
          style={{ background: `radial-gradient(110% 90% at 100% 0%, ${stateColor}1f, transparent 55%)` }}
        />
        <div className="relative flex flex-col gap-8 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-5">
            <div className="relative flex h-16 w-16 items-center justify-center">
              <span
                className="absolute inset-0 rounded-full opacity-30 blur-md"
                style={{ background: stateColor }}
              />
              <span
                className="vital-dot relative h-10 w-10 rounded-full"
                style={{ background: stateColor }}
              />
            </div>
            <div>
              <div className="eyebrow">Current plant state</div>
              <div
                className="mt-1 font-display text-4xl font-light leading-none sm:text-5xl"
                style={{ color: stateColor }}
              >
                {stateLabel}
              </div>
              <div className="mt-2 font-mono text-[11px] text-[var(--sage-dim)]">
                Cluster {stateIdx} · K-means over the moving window
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-x-8 gap-y-5 sm:grid-cols-4">
            <Readout label="Potential" value={latest ? latest.volts.toFixed(3) : "—"} unit="V" />
            <Readout label="Mean" value={latest ? latest.mean.toFixed(3) : "—"} unit="µV" />
            <Readout label="Spikes" value={latest ? latest.spike_count.toFixed(2) : "—"} unit="/win" />
            <Readout label="Hjorth" value={latest ? latest.hjorth.toFixed(3) : "—"} unit="" />
          </div>
        </div>
      </section>

      {/* Instrument array */}
      <div className="mt-6 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Panel
          channel="CH·01"
          title="Membrane Potential"
          hint="mean ± std"
          accent={C_POTENTIAL}
        >
          <ComposedChart data={points} margin={{ top: 6, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" minTickGap={48} tick={AXIS} stroke={GRID} tickLine={false} />
            <YAxis tick={AXIS} stroke={GRID} tickLine={false} width={44} />
            <Tooltip content={<ChartTip unit=" µV" />} cursor={CURSOR} />
            <Area
              dataKey="band"
              stroke="none"
              fill={C_POTENTIAL}
              fillOpacity={0.16}
              isAnimationActive={false}
              name="Std band"
            />
            <Line
              dataKey="mean"
              stroke={C_POTENTIAL}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              name="Mean"
            />
          </ComposedChart>
        </Panel>

        <Panel
          channel="CH·02"
          title="Spike Rate"
          hint="events / window"
          accent={C_SPIKE}
        >
          <LineChart data={points} margin={{ top: 6, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" minTickGap={48} tick={AXIS} stroke={GRID} tickLine={false} />
            <YAxis tick={AXIS} stroke={GRID} tickLine={false} width={44} />
            <Tooltip content={<ChartTip />} cursor={CURSOR} />
            <Line
              dataKey="spike_count"
              stroke={C_SPIKE}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              name="Spikes"
            />
          </LineChart>
        </Panel>

        <Panel
          channel="CH·03"
          title="Hjorth Complexity"
          hint="signal chaos"
          accent={C_HJORTH}
        >
          <LineChart data={points} margin={{ top: 6, right: 8, left: -12, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" minTickGap={48} tick={AXIS} stroke={GRID} tickLine={false} />
            <YAxis tick={AXIS} stroke={GRID} tickLine={false} width={44} />
            <Tooltip content={<ChartTip />} cursor={CURSOR} />
            <Line
              dataKey="hjorth"
              stroke={C_HJORTH}
              strokeWidth={1.6}
              dot={false}
              isAnimationActive={false}
              name="Hjorth"
            />
          </LineChart>
        </Panel>

        <Panel
          channel="CH·04"
          title="Agitation Cluster"
          hint="stable / slight / high"
          accent={stateColor}
        >
          <ScatterChart margin={{ top: 6, right: 8, left: 8, bottom: 0 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" minTickGap={48} tick={AXIS} stroke={GRID} tickLine={false} />
            <YAxis
              dataKey="cluster"
              type="number"
              domain={[-0.5, 2.5]}
              ticks={[0, 1, 2]}
              tickFormatter={(v: number) => CLUSTER_LABELS[v] ?? String(v)}
              tick={AXIS}
              stroke={GRID}
              tickLine={false}
              width={104}
            />
            <ZAxis range={[50, 50]} />
            <Tooltip
              cursor={CURSOR}
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const p = payload[0]?.payload as Point;
                return (
                  <div className="rounded-lg border border-[var(--line-strong)] bg-[var(--surface-2)] px-3 py-2 font-mono text-[11px] text-[var(--ink)] shadow-xl">
                    <div className="text-[var(--sage)]">{p.t}</div>
                    <div className="mt-1 flex items-center gap-2">
                      <span
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ background: CLUSTER_COLORS[p.cluster] }}
                      />
                      {CLUSTER_LABELS[p.cluster] ?? p.cluster}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter data={points} isAnimationActive={false}>
              {points.map((p, i) => (
                <Cell key={i} fill={CLUSTER_COLORS[p.cluster] ?? "#888"} />
              ))}
            </Scatter>
          </ScatterChart>
        </Panel>
      </div>

      <footer className="mt-8 flex items-center justify-between font-mono text-[10px] uppercase tracking-[0.18em] text-[var(--sage-dim)]">
        <span>Plantcasso · ESP32 K-means</span>
        <span>{points.length}/{MAX_POINTS} samples buffered</span>
      </footer>
    </div>
  );
}

// A single instrument readout in the hero strip — mono number, sage eyebrow.
function Readout({
  label,
  value,
  unit,
}: {
  label: string;
  value: string;
  unit: string;
}) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div className="mt-1 font-mono text-2xl tabular-nums text-[var(--ink)]">
        {value}
        {unit && (
          <span className="ml-1 text-sm text-[var(--sage-dim)]">{unit}</span>
        )}
      </div>
    </div>
  );
}
