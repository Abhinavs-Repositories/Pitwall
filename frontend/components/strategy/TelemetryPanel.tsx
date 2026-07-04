"use client";

import type { RaceState, StrategyRecommendation } from "@/lib/types";
import { COMPOUND_COLOR, COMPOUND_LETTER, formatGap, teamColor } from "@/lib/constants";

export default function TelemetryPanel({
  raceState,
  loading,
  latestRecommendation,
}: {
  raceState: RaceState | undefined;
  loading: boolean;
  latestRecommendation: StrategyRecommendation | null;
}) {
  const drivers = [...(raceState?.drivers ?? [])].sort((a, b) => a.position - b.position);

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-b border-border-dim bg-carbon-2 lg:border-b-0 lg:border-r">
      <div className="border-b border-border-dim px-4 py-3">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-text-muted">
          // running order
        </p>
      </div>

      <div className="flex-1 overflow-y-auto">
        {loading && (
          <p className="p-4 font-mono text-xs text-text-muted">Loading race state…</p>
        )}
        {!loading && drivers.length === 0 && (
          <p className="p-4 font-mono text-xs text-text-muted">No driver data for this lap.</p>
        )}
        {drivers.map((d) => (
          <div
            key={d.driver_number}
            className="flex items-center gap-2 border-b border-border-dim px-4 py-2.5 text-xs"
          >
            <span className="w-5 font-mono text-text-secondary">P{d.position}</span>
            <span
              className="h-2 w-2 flex-shrink-0 rounded-full"
              style={{ background: teamColor(d.team) }}
            />
            <div className="min-w-0 flex-1">
              <p className="truncate font-medium text-text-primary">{d.name}</p>
              <p className="truncate font-mono text-[10px] text-text-muted">{d.team}</p>
            </div>
            <span
              className="flex h-5 w-5 flex-shrink-0 items-center justify-center rounded-full font-mono text-[10px] font-bold text-carbon"
              style={{ background: COMPOUND_COLOR[d.tire_compound] }}
              title={d.tire_compound}
            >
              {COMPOUND_LETTER[d.tire_compound]}
            </span>
            <span className="w-16 flex-shrink-0 text-right font-mono text-[11px] text-text-secondary">
              {formatGap(d.gap_to_leader)}
            </span>
          </div>
        ))}
      </div>

      {latestRecommendation && (
        <div className="border-t border-border-accent bg-carbon-3 p-4">
          <p className="mb-1 font-mono text-[10px] uppercase tracking-[0.2em] text-text-muted">
            // latest call · car #{latestRecommendation.driver_number}
          </p>
          <p className="font-display text-sm font-bold text-text-primary">
            {latestRecommendation.recommended_action.replace(/_/g, " ")}
          </p>
          {latestRecommendation.optimal_pit_window && (
            <p className="mt-1 font-mono text-[11px] text-text-secondary">
              Window: L{latestRecommendation.optimal_pit_window[0]}–
              {latestRecommendation.optimal_pit_window[1]}
            </p>
          )}
        </div>
      )}
    </aside>
  );
}
