const RUNNING_ORDER = [
  { code: "VER", color: "var(--team-redbull)" },
  { code: "NOR", color: "var(--team-mclaren)" },
  { code: "LEC", color: "var(--team-ferrari)" },
  { code: "RUS", color: "var(--team-mercedes)" },
  { code: "ALO", color: "var(--team-aston)" },
];

/**
 * Static, non-interactive mockup of the real strategy console for the
 * Section 4 split-screen preview. The functional /strategy page (wired to
 * the live chat/telemetry backend) comes in a later phase.
 */
export default function ConsolePreview() {
  return (
    <div className="w-full max-w-md rounded-lg border border-border-dim bg-carbon-3 p-5 font-mono text-xs shadow-2xl">
      <div className="mb-4 flex items-center justify-between border-b border-border-dim pb-3">
        <span className="text-[10px] uppercase tracking-[0.25em] text-text-muted">
          // strategy console
        </span>
        <span className="flex items-center gap-1.5 text-[10px] uppercase text-data-green">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-data-green" />
          Live
        </span>
      </div>

      <div className="mb-4 flex gap-1.5 overflow-hidden">
        {RUNNING_ORDER.map((d) => (
          <div
            key={d.code}
            className="flex items-center gap-1.5 rounded bg-carbon-4 px-2 py-1 text-text-secondary"
          >
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: d.color }} />
            {d.code}
          </div>
        ))}
      </div>

      <div className="mb-4 space-y-2">
        <p className="ml-auto max-w-[85%] rounded-lg bg-carbon-4 px-3 py-2 text-right text-text-secondary">
          Should VER pit now?
        </p>
        <p className="max-w-[90%] rounded-lg border border-border-dim bg-carbon px-3 py-2 text-text-primary">
          Tyre deg climbing at 0.41s/lap. Cliff projected lap 34. Recommend pitting
          within 2 laps.
        </p>
      </div>

      <div className="rounded-lg border-t-2 border-f1-red bg-carbon-4 p-3">
        <div className="flex items-center justify-between">
          <span className="font-display text-sm font-bold text-text-primary">PIT NOW</span>
          <span className="text-data-green">87% confidence</span>
        </div>
        <div className="mt-1.5 flex gap-3 text-text-muted">
          <span>→ MEDIUM</span>
          <span className="text-data-blue">UNDERCUT VIABLE</span>
        </div>
      </div>
    </div>
  );
}
