"use client";

import { useEffect, useRef } from "react";

interface Streak {
  x: number;
  y: number;
  len: number;
  speed: number;
  width: number;
  color: string;
  opacity: number;
}

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// Weighted toward white (the dominant "speed" streak) with occasional
// glints in real 2026 team colours — the grid blurring past, not a
// single car.
const STREAK_PALETTE: { rgb: string; weight: number }[] = [
  { rgb: "245,245,245", weight: 0.56 }, // white
  { rgb: "232,0,32", weight: 0.16 }, // ferrari red
  { rgb: "54,113,198", weight: 0.12 }, // red bull navy
  { rgb: "255,128,0", weight: 0.09 }, // mclaren papaya
  { rgb: "39,244,210", weight: 0.07 }, // mercedes teal
];

function pickColor(rand: () => number): string {
  const roll = rand();
  let acc = 0;
  for (const { rgb, weight } of STREAK_PALETTE) {
    acc += weight;
    if (roll <= acc) return rgb;
  }
  return STREAK_PALETTE[0].rgb;
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
      color: pickColor(rand),
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
        if (s.x + s.len < 0) {
          s.x = width + rand() * 200;
          s.color = pickColor(rand);
        }

        const y = s.y * height;
        const grad = ctx.createLinearGradient(s.x, y, s.x + s.len, y);
        grad.addColorStop(0, `rgba(${s.color},0)`);
        grad.addColorStop(0.5, `rgba(${s.color},${s.opacity})`);
        grad.addColorStop(1, `rgba(${s.color},0)`);

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
            "radial-gradient(55% 45% at 25% 30%, rgba(232,0,32,0.10), transparent 70%), " +
            "radial-gradient(50% 45% at 80% 85%, rgba(54,113,198,0.08), transparent 70%)",
        }}
      />
      <div className="grain-overlay" />
    </div>
  );
}
