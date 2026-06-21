/**
 * GET /api/stream — Server-Sent Events endpoint.
 *
 * Reads from the ESP32 serial port *on the server* (Node runtime) and streams
 * each normalised reading to the browser as an SSE `data:` frame. The client
 * consumes this with `EventSource`.
 */

import type { NextRequest } from "next/server";
import { getSerialReader, type Reading } from "@/app/lib/serial";

// Serial access requires the Node runtime, and the stream must never be cached
// or statically rendered.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
  const encoder = new TextEncoder();
  const reader = getSerialReader();

  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      const send = (r: Reading) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(r)}\n\n`));
        } catch {
          closed = true;
        }
      };

      // Push the most recent reading immediately so a fresh client isn't blank.
      const latest = reader.getLatest();
      if (latest) send(latest);

      const unsubscribe = reader.subscribe(send);

      // Heartbeat keeps proxies/browsers from dropping an idle connection.
      const heartbeat = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(": ping\n\n"));
        } catch {
          closed = true;
        }
      }, 15000);

      const cleanup = () => {
        if (closed) return;
        closed = true;
        clearInterval(heartbeat);
        unsubscribe();
        try {
          controller.close();
        } catch {
          // already closed
        }
      };

      request.signal.addEventListener("abort", cleanup);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
