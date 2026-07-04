"use client";

import { useEffect, useRef } from "react";

interface Streak {
  x: number;
  y: number;
  len: number;
  speed: number;
  width: number;
  hue: "white" | "red";
  opacity: number;
}

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/**
 * Procedural stand-in for "high-contrast race photography": a long-exposure
 * light-streak field drawn on a 2D canvas with a persistence trail. No
 * stock imagery required, and it reads as speed/motion at a glance.
 */
export default function SpeedStreakCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let width = 0;
    let height = 0;
    const rand = seededRandom(42);
    const streaks: Streak[] = Array.from({ length: 46 }, () => ({
      x: rand() * 2000,
      y: rand(),
      len: 120 + rand() * 260,
      speed: 2.2 + rand() * 4.2,
      width: 0.6 + rand() * 1.8,
      hue: rand() > 0.86 ? "red" : "white",
      opacity: 0.08 + rand() * 0.22,
    }));

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = canvas.width = rect.width * dpr;
      height = canvas.height = rect.height * dpr;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      ctx.fillStyle = "rgba(10, 10, 10, 0.18)";
      ctx.fillRect(0, 0, width, height);

      for (const s of streaks) {
        s.x -= s.speed;
        if (s.x + s.len < 0) s.x = width + rand() * 200;

        const y = s.y * height;
        const grad = ctx.createLinearGradient(s.x, y, s.x + s.len, y);
        const color = s.hue === "red" ? "232,0,45" : "245,245,245";
        grad.addColorStop(0, `rgba(${color},0)`);
        grad.addColorStop(0.5, `rgba(${color},${s.opacity})`);
        grad.addColorStop(1, `rgba(${color},0)`);

        ctx.strokeStyle = grad;
        ctx.lineWidth = s.width;
        ctx.beginPath();
        ctx.moveTo(s.x, y);
        ctx.lineTo(s.x + s.len, y);
        ctx.stroke();
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div className="relative h-full w-full overflow-hidden bg-carbon">
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(60% 50% at 30% 40%, rgba(232,0,45,0.10), transparent 70%)",
        }}
      />
      <div className="grain-overlay" />
    </div>
  );
}
