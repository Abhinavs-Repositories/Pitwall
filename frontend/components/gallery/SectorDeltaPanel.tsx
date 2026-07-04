const DELTA_COLOR: Record<string, string> = {
  purple: "#b967ff",
  green: "var(--data-green)",
  yellow: "var(--data-amber)",
};

const SECTORS = [
  { label: "S1", time: "28.417", delta: "purple", fill: 0.92 },
  { label: "S2", time: "41.902", delta: "green", fill: 0.7 },
  { label: "S3", time: "25.114", delta: "yellow", fill: 0.55 },
];

export default function SectorDeltaPanel() {
  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // sector deltas
      </p>
      <div className="flex flex-col gap-4 py-4">
        {SECTORS.map((s) => (
          <div key={s.label} className="flex items-center gap-3">
            <span className="w-6 font-mono text-xs text-text-secondary">{s.label}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-carbon-4">
              <div
                className="h-full rounded-full"
                style={{ width: `${s.fill * 100}%`, background: DELTA_COLOR[s.delta] }}
              />
            </div>
            <span className="font-mono text-xs" style={{ color: DELTA_COLOR[s.delta] }}>
              {s.time}
            </span>
          </div>
        ))}
      </div>
      <p className="font-display text-xl font-bold text-text-primary">Live Sector Splits</p>
    </div>
  );
}
