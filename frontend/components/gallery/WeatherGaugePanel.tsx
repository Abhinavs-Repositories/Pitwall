const GAUGE_RADIUS = 40;
const FILL_FRACTION = 0.68;

export default function WeatherGaugePanel() {
  const circumference = 2 * Math.PI * GAUGE_RADIUS;

  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // track conditions
      </p>
      <div className="flex items-center justify-center py-2">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r={GAUGE_RADIUS} fill="none" stroke="var(--carbon-4)" strokeWidth="8" />
          <circle
            cx="50"
            cy="50"
            r={GAUGE_RADIUS}
            fill="none"
            stroke="var(--data-amber)"
            strokeWidth="8"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - FILL_FRACTION)}
            strokeLinecap="round"
            transform="rotate(-90 50 50)"
          />
          <text x="50" y="47" textAnchor="middle" fontSize="18" fill="var(--text-primary)" fontWeight={700}>
            34°
          </text>
          <text x="50" y="63" textAnchor="middle" fontSize="8" fill="var(--text-muted)">
            TRACK C
          </text>
        </svg>
      </div>
      <p className="font-display text-xl font-bold text-text-primary">Weather Grid</p>
    </div>
  );
}
