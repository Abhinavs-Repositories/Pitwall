"use client";

export default function LapControl({
  lap,
  totalLaps,
  onChange,
}: {
  lap: number;
  totalLaps: number;
  onChange: (lap: number) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-xs text-text-muted">LAP</span>
      <input
        type="range"
        min={1}
        max={Math.max(totalLaps, 1)}
        value={lap}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-40 accent-f1-red"
      />
      <span className="w-14 font-mono text-xs text-text-primary">
        {lap} / {totalLaps}
      </span>
    </div>
  );
}
