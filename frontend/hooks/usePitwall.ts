"use client";

import { useCallback, useRef, useState } from "react";
import { streamChat } from "@/lib/api";
import type { ChatMessage, StrategyRecommendation } from "@/lib/types";

export interface ChatEntry {
  role: "user" | "assistant";
  content: string;
  strategyData?: StrategyRecommendation | null;
  agentsUsed?: string[];
  pending?: boolean;
}

export function usePitwall(sessionKey: number | null, currentLap: number) {
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const historyRef = useRef<ChatMessage[]>([]);

  const send = useCallback(
    async (userMessage: string) => {
      if (!sessionKey || isStreaming || !userMessage.trim()) return;

      const history = historyRef.current;
      setMessages((prev) => [
        ...prev,
        { role: "user", content: userMessage },
        { role: "assistant", content: "", pending: true },
      ]);
      setIsStreaming(true);

      let text = "";
      let strategyData: StrategyRecommendation | null = null;
      let agentsUsed: string[] = [];

      const updateLast = (entry: ChatEntry) => {
        setMessages((prev) => {
          const next = [...prev];
          next[next.length - 1] = entry;
          return next;
        });
      };

      try {
        for await (const evt of streamChat({
          session_key: sessionKey,
          current_lap: currentLap,
          message: userMessage,
          conversation_history: history,
        })) {
          if (evt.event === "meta") {
            const meta = JSON.parse(evt.data);
            strategyData = meta.strategy_data ?? null;
            agentsUsed = meta.agents_used ?? [];
          } else if (evt.event === "token") {
            text += evt.data;
            updateLast({ role: "assistant", content: text, strategyData, agentsUsed, pending: true });
          } else if (evt.event === "error") {
            text = `Error: ${evt.data}`;
          }
        }

        updateLast({ role: "assistant", content: text, strategyData, agentsUsed, pending: false });
        historyRef.current = [
          ...history,
          { role: "user", content: userMessage },
          { role: "assistant", content: text },
        ].slice(-10);
      } catch (err) {
        updateLast({ role: "assistant", content: `Error: ${(err as Error).message}`, pending: false });
      } finally {
        setIsStreaming(false);
      }
    },
    [sessionKey, currentLap, isStreaming]
  );

  return { messages, send, isStreaming };
}
