import type { ChatRequest, ChatResponse, RaceState, RaceSummaryListing } from "./types";

const BASE = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000") + "/api";

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} → ${res.status}: ${body}`);
  }
  return res.json();
}

export function getRaces(year?: number): Promise<RaceSummaryListing[]> {
  return getJSON(`/races${year ? `?year=${year}` : ""}`);
}

export function getRaceState(sessionKey: number): Promise<RaceState> {
  return getJSON(`/races/${sessionKey}`);
}

export function getRaceStateAtLap(sessionKey: number, lap: number): Promise<RaceState> {
  return getJSON(`/races/${sessionKey}/lap/${lap}`);
}

export function getRaceSummary(
  sessionKey: number,
  lap: number
): Promise<{ summary: string; processing_time_ms: string }> {
  return getJSON(`/races/${sessionKey}/summary/${lap}`);
}

export async function sendChatMessage(payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Chat API error: ${res.status}`);
  return res.json();
}

export interface StreamEvent {
  event: string;
  data: string;
}

/**
 * Parses the backend's SSE stream (event/data blocks separated by a blank
 * line). Multi-line `data:` fields — used for tokens containing literal
 * newlines — are rejoined with "\n" per the SSE spec rather than
 * concatenated flat.
 */
export async function* streamChat(payload: ChatRequest): AsyncGenerator<StreamEvent> {
  const res = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok || !res.body) throw new Error(`Chat stream error: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);

      let eventType = "message";
      const dataLines: string[] = [];
      for (const line of rawEvent.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
      }
      if (dataLines.length > 0) {
        yield { event: eventType, data: dataLines.join("\n") };
      }
    }
  }
}
