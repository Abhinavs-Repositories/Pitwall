"use client";

import { useEffect, useRef } from "react";

interface RadialStreak {
  angle: number;
  dist: number;
  speed: number;
  color: string;
}

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

const STREAK_COLORS = ["#e80020", "#3671c6", "#ff8000", "#27f4d2", "#ffffff", "#ffffff"];

/**
 * Procedural cockpit-at-speed sensation for the "On Track" panel — radial
 * streaks rushing outward from a vanishing point, standing in for race
 * photography.
 */
export default function OnTrackVisual() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let width = 0;
    let height = 0;
    const rand = seededRandom(77);

    const makeStreak = (): RadialStreak => ({
      angle: rand() * Math.PI * 2,
      dist: rand() * 60,
      speed: 3.5 + rand() * 5,
      color: STREAK_COLORS[Math.floor(rand() * STREAK_COLORS.length)],
    });

    const streaks: RadialStreak[] = Array.from({ length: 70 }, makeStreak);

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      width = canvas.width = rect.width * dpr;
      height = canvas.height = rect.height * dpr;
    };
    resize();
    window.addEventListener("resize", resize);

    const draw = () => {
      ctx.fillStyle = "rgba(10, 10, 10, 0.22)";
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const maxR = Math.hypot(width, height) / 2;

      for (const s of streaks) {
        const innerR = s.dist;
        const outerR = s.dist + 40 + s.speed * 3;
        const x1 = cx + Math.cos(s.angle) * innerR;
        const y1 = cy + Math.sin(s.angle) * innerR;
        const x2 = cx + Math.cos(s.angle) * outerR;
        const y2 = cy + Math.sin(s.angle) * outerR;

        const fade = Math.min(1, s.dist / (maxR * 0.5));
        const opacity = 0.5 * (1 - fade * 0.7);

        ctx.strokeStyle = s.color;
        ctx.globalAlpha = Math.max(0, opacity);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.globalAlpha = 1;

        s.dist += s.speed;
        if (s.dist > maxR) {
          s.angle = rand() * Math.PI * 2;
          s.dist = 0;
          s.speed = 3.5 + rand() * 5;
          s.color = STREAK_COLORS[Math.floor(rand() * STREAK_COLORS.length)];
        }
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
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(35% 35% at 50% 50%, rgba(0,0,0,0), rgba(10,10,10,0.85) 75%)",
        }}
      />
      <div className="grain-overlay" />
    </div>
  );
}
