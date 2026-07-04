// A different generative circuit sketch from the hero's 3D wireframe —
// a flat, angular top-down layout with numbered corner markers.
const TRACK_POINTS: [number, number][] = [
  [16, 132],
  [144, 132],
  [176, 96],
  [150, 40],
  [92, 28],
  [60, 60],
  [16, 96],
  [16, 132],
];

const CORNER_MARKERS: { at: [number, number]; n: number }[] = [
  { at: [16, 132], n: 1 },
  { at: [176, 96], n: 2 },
  { at: [92, 28], n: 3 },
  { at: [16, 96], n: 4 },
];

export default function CircuitMapPanel() {
  const d = TRACK_POINTS.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ");

  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // circuit map
      </p>
      <svg viewBox="0 0 192 160" className="w-full flex-1">
        <path
          d={d}
          fill="none"
          stroke="#ffffff"
          strokeOpacity={0.35}
          strokeWidth={3}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {CORNER_MARKERS.map(({ at, n }) => (
          <g key={n}>
            <circle cx={at[0]} cy={at[1]} r={9} fill="#e8002d" />
            <text x={at[0]} y={at[1] + 3} textAnchor="middle" fontSize={9} fill="#fff">
              {n}
            </text>
          </g>
        ))}
      </svg>
      <p className="font-display text-xl font-bold text-text-primary">Track Characteristics</p>
    </div>
  );
}
