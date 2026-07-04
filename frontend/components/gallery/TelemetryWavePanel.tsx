"use client";

import { useEffect, useRef } from "react";

/** Animated oscilloscope-style trace standing in for a live throttle signal. */
export default function TelemetryWavePanel() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let width = 0;
    let height = 0;
    let t = 0;

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = canvas.width = rect.width * dpr;
      height = canvas.height = rect.height * dpr;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      t += 0.02;
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = "#00d26a";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let x = 0; x <= width; x += 4) {
        const nx = x / width;
        const y =
          height / 2 +
          Math.sin(nx * 14 + t) * height * 0.22 * Math.sin(nx * 3 + t * 0.5) +
          Math.sin(nx * 40 + t * 3) * height * 0.03;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // throttle trace
      </p>
      <canvas ref={canvasRef} className="h-24 w-full" />
      <p className="font-display text-xl font-bold text-text-primary">Live Telemetry</p>
    </div>
  );
}
