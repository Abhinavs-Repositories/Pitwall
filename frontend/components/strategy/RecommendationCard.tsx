"use client";

import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import type { StrategyRecommendation } from "@/lib/types";
import { COMPOUND_COLOR } from "@/lib/constants";

function actionColor(action: string): string {
  if (action.startsWith("PIT_NOW")) return "var(--f1-red)";
  if (action.startsWith("STAY_OUT")) return "var(--data-green)";
  if (action.startsWith("PIT_IN")) return "var(--data-amber)";
  return "var(--text-secondary)";
}

export default function RecommendationCard({ data }: { data: StrategyRecommendation }) {
  const borderRef = useRef<HTMLDivElement>(null);

  useGSAP(() => {
    gsap.fromTo(
      borderRef.current,
      { scaleX: 0 },
      { scaleX: 1, duration: 0.4, ease: "power2.out" }
    );
  }, []);

  const color = actionColor(data.recommended_action);

  return (
    <div className="relative overflow-hidden rounded-lg border border-border-dim bg-carbon-3 p-4">
      <div
        ref={borderRef}
        className="absolute left-0 top-0 h-[3px] w-full origin-left"
        style={{ background: color }}
      />
      <div className="flex items-center justify-between gap-3">
        <span className="font-display text-lg font-bold uppercase" style={{ color }}>
          {data.recommended_action.replace(/_/g, " ")}
        </span>
        <span className="whitespace-nowrap font-mono text-xs text-text-secondary">
          {Math.round(data.confidence * 100)}% confidence
        </span>
      </div>

      <div className="mt-2 flex flex-wrap gap-3 font-mono text-xs text-text-secondary">
        {data.recommended_compound && (
          <span className="flex items-center gap-1.5">
            <span
              className="h-2 w-2 rounded-full"
              style={{ background: COMPOUND_COLOR[data.recommended_compound] }}
            />
            {data.recommended_compound}
          </span>
        )}
        {data.optimal_pit_window && (
          <span>
            Window: L{data.optimal_pit_window[0]}–{data.optimal_pit_window[1]}
          </span>
        )}
        {data.undercut_viable && <span className="text-data-blue">UNDERCUT VIABLE</span>}
        {data.overcut_viable && <span className="text-data-blue">OVERCUT VIABLE</span>}
      </div>

      <p className="mt-3 text-xs leading-relaxed text-text-secondary">{data.reasoning}</p>
    </div>
  );
}
