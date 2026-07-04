"use client";

import { useEffect, useRef, useState } from "react";
import type { ChatEntry } from "@/hooks/usePitwall";
import type { RaceState } from "@/lib/types";
import RecommendationCard from "./RecommendationCard";
import AgentTrace from "./AgentTrace";
import Chip from "@/components/ui/Chip";

const QUICK_CHIPS = [
  "Should the leader pit now?",
  "What's the tyre degradation looking like?",
  "Any undercut opportunities?",
  "What's the weather forecast?",
];

/** The explainer LLM writes **bold** markdown — render just that, not a full parser. */
function formatBold(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i}>{part.slice(2, -2)}</strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

export default function StrategyChat({
  messages,
  onSend,
  isStreaming,
  raceState,
}: {
  messages: ChatEntry[];
  onSend: (msg: string) => void;
  isStreaming: boolean;
  raceState: RaceState | undefined;
}) {
  const [input, setInput] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  function submit() {
    if (!input.trim()) return;
    onSend(input);
    setInput("");
  }

  return (
    <section className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-carbon">
      <div ref={scrollRef} className="flex-1 space-y-4 overflow-y-auto p-6">
        {messages.length === 0 && (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center">
            <p className="font-display text-2xl font-bold uppercase text-text-primary">
              Ask the pit wall
            </p>
            <p className="max-w-sm font-mono text-xs text-text-muted">
              &quot;Should Verstappen pit now?&quot; · &quot;Compare Norris vs Leclerc tyres&quot; ·
              &quot;What happened at this track before?&quot;
            </p>
          </div>
        )}

        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <p className="max-w-[75%] rounded-lg bg-carbon-3 px-4 py-2.5 text-sm text-text-primary">
                {m.content}
              </p>
            </div>
          ) : (
            <div key={i} className="flex flex-col gap-2">
              <div className="max-w-[85%] space-y-2">
                <p className="rounded-lg border border-border-dim bg-carbon-2 px-4 py-2.5 text-sm text-text-primary">
                  {m.content ? formatBold(m.content) : m.pending ? "…" : ""}
                </p>
                {m.agentsUsed && m.agentsUsed.length > 0 && !m.pending && (
                  <AgentTrace agents={m.agentsUsed} />
                )}
                {m.strategyData && <RecommendationCard data={m.strategyData} />}
              </div>
            </div>
          )
        )}
      </div>

      <div className="border-t border-border-dim p-4">
        <div className="mb-3 flex flex-wrap gap-2">
          {QUICK_CHIPS.map((c) => (
            <Chip key={c} onClick={() => onSend(c)}>
              {c}
            </Chip>
          ))}
        </div>
        <div className="flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            placeholder={raceState ? "Ask a strategy question…" : "Select a race first"}
            disabled={!raceState || isStreaming}
            className="flex-1 rounded-md border border-border-dim bg-carbon-3 px-4 py-2.5 text-sm text-text-primary outline-none placeholder:text-text-muted focus:border-border-accent disabled:opacity-50"
          />
          <button
            onClick={submit}
            disabled={!raceState || isStreaming || !input.trim()}
            className="rounded-md bg-f1-red px-5 py-2.5 font-mono text-xs uppercase text-text-primary transition-opacity disabled:opacity-40"
          >
            Send
          </button>
        </div>
      </div>
    </section>
  );
}
