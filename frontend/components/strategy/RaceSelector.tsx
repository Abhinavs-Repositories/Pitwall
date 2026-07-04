"use client";

import type { RaceSummaryListing } from "@/lib/types";

export default function RaceSelector({
  races,
  value,
  onChange,
}: {
  races: RaceSummaryListing[];
  value: number | null;
  onChange: (sessionKey: number) => void;
}) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(Number(e.target.value))}
      className="rounded-md border border-border-dim bg-carbon-3 px-3 py-2 font-mono text-xs text-text-primary outline-none focus:border-border-accent"
    >
      {races.length === 0 && <option value="">Loading races…</option>}
      {races.map((r) => (
        <option key={r.session_key} value={r.session_key}>
          {r.year} — {r.circuit_short_name || r.location} ({r.country_name})
        </option>
      ))}
    </select>
  );
}
