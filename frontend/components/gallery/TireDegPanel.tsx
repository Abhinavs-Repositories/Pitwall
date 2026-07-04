// Mirrors the shape of the real tire_deg.py output: a near-linear climb
// in lap time followed by a cliff once degradation outruns the compound.
const LAP_COUNT = 16;
const CLIFF_START_LAP = 12;

function buildLapTimes() {
  return Array.from({ length: LAP_COUNT }, (_, i) => {
    const base = 92.0 + i * 0.18;
    const cliff = i > CLIFF_START_LAP ? (i - CLIFF_START_LAP) * 0.9 : 0;
    return base + cliff;
  });
}

export default function TireDegPanel() {
  const laps = buildLapTimes();
  const max = Math.max(...laps);
  const min = Math.min(...laps);
  const points = laps
    .map((t, i) => {
      const x = (i / (laps.length - 1)) * 100;
      const y = 100 - ((t - min) / (max - min)) * 100;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const cliffX = (CLIFF_START_LAP / (laps.length - 1)) * 100;

  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // tyre degradation
      </p>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-32 w-full">
        <rect x={cliffX} y="0" width={100 - cliffX} height="100" fill="var(--f1-red)" opacity="0.08" />
        <polyline points={points} fill="none" stroke="var(--data-amber)" strokeWidth="1.5" />
      </svg>
      <div className="flex items-baseline justify-between">
        <p className="font-display text-xl font-bold text-text-primary">Cliff Prediction</p>
        <span className="font-mono text-xs text-f1-red">+2.5s / lap {CLIFF_START_LAP + 1}</span>
      </div>
    </div>
  );
}
