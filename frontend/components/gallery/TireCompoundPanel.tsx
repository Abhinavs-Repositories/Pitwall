const COMPOUNDS = [
  { label: "SOFT", short: "S", color: "var(--compound-soft)" },
  { label: "MEDIUM", short: "M", color: "var(--compound-medium)" },
  { label: "HARD", short: "H", color: "var(--compound-hard)" },
];

export default function TireCompoundPanel() {
  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // tyre compounds
      </p>
      <div className="flex items-end justify-center gap-6 py-4">
        {COMPOUNDS.map((c) => (
          <div key={c.label} className="flex flex-col items-center gap-2">
            <svg width="52" height="52" viewBox="0 0 52 52">
              <circle cx="26" cy="26" r="22" fill="none" stroke={c.color} strokeWidth="4" opacity={0.9} />
              <circle cx="26" cy="26" r="13" fill="none" stroke={c.color} strokeWidth="2" opacity={0.4} />
              {Array.from({ length: 12 }).map((_, i) => {
                const angle = (i / 12) * Math.PI * 2;
                // Fixed precision avoids an SSR/client hydration mismatch —
                // Math.cos/sin can differ in the last float digit between
                // Node's V8 and the browser's.
                const x1 = (26 + Math.cos(angle) * 17).toFixed(2);
                const y1 = (26 + Math.sin(angle) * 17).toFixed(2);
                const x2 = (26 + Math.cos(angle) * 22).toFixed(2);
                const y2 = (26 + Math.sin(angle) * 22).toFixed(2);
                return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={c.color} strokeWidth="2" />;
              })}
            </svg>
            <span className="font-mono text-[10px] text-text-secondary">{c.short}</span>
          </div>
        ))}
      </div>
      <p className="font-display text-xl font-bold text-text-primary">Compound Strategy</p>
    </div>
  );
}
