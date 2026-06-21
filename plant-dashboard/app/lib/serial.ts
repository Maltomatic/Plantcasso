/**
 * serial.ts — server-side serial reader (Node runtime only)
 * ==========================================================
 * Opens the ESP32 serial port, parses the newline-delimited JSON emitted by
 * `plant_inference.ino`, normalises it to the dashboard's `Reading` shape, and
 * fans it out to any subscribers (the SSE route handler).
 *
 * ESP32 emits one JSON object per inference window, e.g.
 *   {"cluster":0,"servo":[166,98,10,33,75],"mean":-0.0,"spk":0.02,"chaos":1.382,"volts":0.1725}
 *
 * If the serial port cannot be opened (no device attached, dev machine, …) we
 * fall back to emitting simulated data so the dashboard stays usable — this
 * mirrors the behaviour of the original Flask `dashboard-2/app.py`.
 *
 * A single instance is cached on `globalThis` so Next.js dev HMR / multiple
 * route invocations don't open the port more than once.
 */

import { SerialPort, ReadlineParser } from "serialport";

export interface Reading {
  timestamp: number; // epoch seconds (matches original Flask payload)
  mean: number;
  std: number;
  spike_count: number;
  hjorth: number;
  cluster: number;
  volts: number;
}

type Listener = (r: Reading) => void;

const SERIAL_PORT = process.env.SERIAL_PORT ?? "/dev/ttyUSB0";
const SERIAL_BAUD = Number(process.env.SERIAL_BAUD ?? 115200);

/** Raw JSON object shape emitted by plant_inference.ino. */
interface Esp32Json {
  cluster?: number;
  servo?: number[];
  mean?: number;
  spk?: number;
  chaos?: number;
  volts?: number;
}

/**
 * Map the firmware JSON onto the dashboard schema. The firmware no longer emits
 * a separate `std`, so — preserving the original dashboard's behaviour — the
 * spike rate (`spk`) drives both the std band and the spike-count chart, while
 * `chaos` (Hjorth complexity) drives the Hjorth chart.
 */
function normalise(j: Esp32Json): Reading {
  const spk = Number(j.spk ?? 0);
  return {
    timestamp: Date.now() / 1000,
    mean: Number(j.mean ?? 0),
    std: spk,
    spike_count: spk,
    hjorth: Number(j.chaos ?? 0),
    cluster: Number(j.cluster ?? 0),
    volts: Number(j.volts ?? 0),
  };
}

class SerialReader {
  private listeners = new Set<Listener>();
  private latest: Reading | null = null;
  private started = false;
  private simTimer: NodeJS.Timeout | null = null;

  /** Lazily open the port / start the simulator on first subscriber. */
  private ensureStarted(): void {
    if (this.started) return;
    this.started = true;

    try {
      const port = new SerialPort({
        path: SERIAL_PORT,
        baudRate: SERIAL_BAUD,
      });
      const parser = port.pipe(new ReadlineParser({ delimiter: "\n" }));

      port.on("open", () =>
        console.log(`[serial] connected on ${SERIAL_PORT} @ ${SERIAL_BAUD}`)
      );
      parser.on("data", (line: string) => this.handleLine(line));
      port.on("error", (err) => {
        console.error(`[serial] error: ${err.message} — using simulation`);
        this.startSimulation();
      });
    } catch (err) {
      console.error(
        `[serial] could not open ${SERIAL_PORT}: ${(err as Error).message} — using simulation`
      );
      this.startSimulation();
    }
  }

  private handleLine(line: string): void {
    const trimmed = line.trim();
    if (!trimmed.startsWith("{")) return; // skip boot/info log lines
    try {
      this.emit(normalise(JSON.parse(trimmed) as Esp32Json));
    } catch {
      // ignore malformed / partial lines
    }
  }

  /** Emit simulated readings (~2 Hz) when no hardware is available. */
  private startSimulation(): void {
    if (this.simTimer) return;
    this.simTimer = setInterval(() => {
      const spike = Math.floor(Math.random() * 4);
      this.emit({
        timestamp: Date.now() / 1000,
        mean: 1.5 + (Math.random() - 0.5) * 0.2,
        std: 0.01 + Math.random() * 0.02,
        spike_count: spike,
        hjorth: 0.7 + Math.random() * 0.4,
        cluster: Math.floor(Math.random() * 3),
        volts: 1.5 + (Math.random() - 0.5) * 0.4,
      });
    }, 500);
  }

  private emit(r: Reading): void {
    this.latest = r;
    for (const l of this.listeners) l(r);
  }

  getLatest(): Reading | null {
    return this.latest;
  }

  subscribe(fn: Listener): () => void {
    this.ensureStarted();
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

// Cache on globalThis so HMR / repeated imports reuse the same open port.
const globalForSerial = globalThis as unknown as {
  __plantSerialReader?: SerialReader;
};

export function getSerialReader(): SerialReader {
  return (globalForSerial.__plantSerialReader ??= new SerialReader());
}
