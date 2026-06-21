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

const CLUSTER_COLORS = ["#00C853", "#FF9800", "#FF5252"]; // stable / slight / high
const CLUSTER_LABELS = ["Stable", "Slight Agitation", "High Agitation"];

function fmtTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour12: false,
  });
}

function Panel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md bg-[#4c6b47] p-4">
      <h2 className="mb-2 text-sm font-semibold text-white">{title}</h2>
      <div className="h-72 w-full rounded bg-white p-2">
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

  if (!entered) {
    return (
      <div
        className="fixed inset-0 z-[1000] flex items-center justify-center overflow-hidden bg-white transition-opacity duration-1000"
        style={{ opacity: fading ? 0 : 1 }}
      >
        <video
          autoPlay
          muted
          loop
          className="absolute h-full w-auto object-cover opacity-60"
        >
          <source src="/vid.mp4" type="video/mp4" />
        </video>
        <div className="relative z-10 px-5 text-center text-[#506043]">
          <h1 className="mb-3 text-5xl font-bold">
            Plantcasso: Biosignal Agitation Dashboard
          </h1>
          <p className="mb-8 mt-6 text-3xl">
            Real-time anomaly detection using K-means clustering
          </p>
          <button
            onClick={enterDashboard}
            className="cursor-pointer border-none bg-[#376639] px-5 py-2.5 text-lg text-white hover:bg-[#2c5230]"
          >
            Enter Dashboard
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full p-5">
      <h1 className="mb-6 text-center text-2xl font-bold">
        Plant Biosignal Dashboard
      </h1>

      <div className="mx-auto grid max-w-[1400px] grid-cols-1 gap-5 lg:grid-cols-2">
        <Panel title="Mean ± Std (Moving Window)">
          <ComposedChart data={points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Area
              dataKey="band"
              stroke="none"
              fill="#2196F3"
              fillOpacity={0.2}
              isAnimationActive={false}
              name="Std band"
            />
            <Line
              dataKey="mean"
              stroke="#2196F3"
              dot={false}
              isAnimationActive={false}
              name="Mean"
            />
          </ComposedChart>
        </Panel>

        <Panel title="Spike Count (Current Window)">
          <LineChart data={points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line
              dataKey="spike_count"
              stroke="#FF5722"
              dot={{ r: 2 }}
              isAnimationActive={false}
              name="Spike count"
            />
          </LineChart>
        </Panel>

        <Panel title="Hjorth Complexity">
          <LineChart data={points}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line
              dataKey="hjorth"
              stroke="#00C853"
              dot={{ r: 2 }}
              isAnimationActive={false}
              name="Hjorth complexity"
            />
          </LineChart>
        </Panel>

        <Panel title="Current Cluster (0=stable, 1=slight, 2=high agitation)">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="t" minTickGap={40} tick={{ fontSize: 11 }} />
            <YAxis
              dataKey="cluster"
              type="number"
              domain={[-0.5, 2.5]}
              ticks={[0, 1, 2]}
              tickFormatter={(v: number) => CLUSTER_LABELS[v] ?? String(v)}
              tick={{ fontSize: 11 }}
              width={110}
            />
            <ZAxis range={[60, 60]} />
            <Tooltip />
            <Scatter data={points} isAnimationActive={false}>
              {points.map((p, i) => (
                <Cell key={i} fill={CLUSTER_COLORS[p.cluster] ?? "#888"} />
              ))}
            </Scatter>
          </ScatterChart>
        </Panel>
      </div>
    </div>
  );
}
