"use client";

import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DriverState } from "@/lib/types";

export default function LapTimeChart({ driver }: { driver: DriverState | undefined }) {
  const data = (driver?.lap_times ?? [])
    .filter((l) => l.lap_time !== null)
    .map((l) => ({ lap: l.lap_number, time: l.lap_time as number }));

  if (data.length === 0) {
    return <p className="py-6 font-mono text-xs text-text-muted">No lap time data yet.</p>;
  }

  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
        <CartesianGrid stroke="var(--border-dim)" vertical={false} />
        <XAxis
          dataKey="lap"
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "var(--text-muted)", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          domain={["dataMin - 1", "dataMax + 1"]}
        />
        <Tooltip
          contentStyle={{
            background: "var(--carbon-3)",
            border: "1px solid var(--border-dim)",
            fontSize: 11,
          }}
          labelStyle={{ color: "var(--text-muted)" }}
          formatter={(value: number) => [`${value.toFixed(3)}s`, "Lap time"]}
          labelFormatter={(lap) => `Lap ${lap}`}
        />
        <Line type="monotone" dataKey="time" stroke="var(--f1-red)" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
