"use client";

import { useEffect, useState } from "react";

/** Typewriter reveal of the agent pipeline, e.g. "router → race_state → strategy". */
export default function AgentTrace({ agents }: { agents: string[] }) {
  const fullText = agents.join(" → ");
  const [shown, setShown] = useState("");

  useEffect(() => {
    setShown("");
    let interval: ReturnType<typeof setInterval> | undefined;

    const delay = setTimeout(() => {
      let i = 0;
      interval = setInterval(() => {
        i += 1;
        setShown(fullText.slice(0, i));
        if (i >= fullText.length && interval) clearInterval(interval);
      }, 18);
    }, 200);

    return () => {
      clearTimeout(delay);
      if (interval) clearInterval(interval);
    };
  }, [fullText]);

  if (!agents.length) return null;

  return (
    <p className="font-mono text-[11px] text-text-muted">
      <span>// agents: </span>
      {shown}
      <span className="animate-pulse">▌</span>
    </p>
  );
}
