"use client";

import type { RaceState } from "@/lib/types";
import LapTimeChart from "./LapTimeChart";
import { COMPOUND_COLOR, formatLapTime } from "@/lib/constants";

export default function LapTimePanel({ raceState }: { raceState: RaceState | undefined }) {
  const leader = raceState?.drivers.find((d) => d.position === 1);
  const weather = raceState?.weather;

  return (
    <aside className="flex h-full min-h-0 flex-col overflow-hidden border-t border-border-dim bg-carbon-2 lg:border-l lg:border-t-0">
      <div className="border-b border-border-dim px-4 py-3">
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-text-muted">
          // leader trace
        </p>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="border-b border-border-dim p-4">
          <p className="font-mono text-xs text-text-muted">{leader?.name ?? "—"}</p>
          <p className="font-display text-4xl font-bold text-text-primary">
            {formatLapTime(leader?.last_lap_time ?? null)}
          </p>
          <LapTimeChart driver={leader} />
        </div>

        <div className="border-b border-border-dim p-4">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-text-muted">
            // weather
          </p>
          <div className="grid grid-cols-2 gap-3 font-mono text-xs">
            <div>
              <p className="text-text-muted">Air</p>
              <p className="text-text-primary">{weather?.air_temp?.toFixed(0) ?? "—"}°C</p>
            </div>
            <div>
              <p className="text-text-muted">Track</p>
              <p className="text-text-primary">{weather?.track_temp?.toFixed(0) ?? "—"}°C</p>
            </div>
            <div>
              <p className="text-text-muted">Humidity</p>
              <p className="text-text-primary">{weather?.humidity?.toFixed(0) ?? "—"}%</p>
            </div>
            <div>
              <p className="text-text-muted">Rain</p>
              <p className={weather?.rainfall ? "text-data-blue" : "text-text-primary"}>
                {weather?.rainfall ? "YES" : "No"}
              </p>
            </div>
          </div>
        </div>

        <div className="p-4">
          <p className="mb-2 font-mono text-[11px] uppercase tracking-[0.2em] text-text-muted">
            // stint data — {leader?.name ?? "—"}
          </p>
          <div className="space-y-2">
            {leader?.stints.length ? (
              leader.stints.map((s) => (
                <div
                  key={s.stint_number}
                  className="flex items-center justify-between font-mono text-xs"
                >
                  <span className="flex items-center gap-1.5 text-text-secondary">
                    <span
                      className="h-2 w-2 rounded-full"
                      style={{ background: COMPOUND_COLOR[s.compound] }}
                    />
                    Stint {s.stint_number} · {s.compound}
                  </span>
                  <span className="text-text-secondary">
                    L{s.lap_start}–{s.lap_end}
                  </span>
                </div>
              ))
            ) : (
              <p className="font-mono text-xs text-text-muted">No stint data yet.</p>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}
