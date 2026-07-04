"use client";

import { useEffect, useRef } from "react";

export interface CursorVec {
  x: number;
  y: number;
}

/**
 * Tracks normalised cursor position (-1..1 on each axis) and exposes a
 * `tick(factor)` step function for lerping toward it inside an animation
 * loop (e.g. R3F's useFrame). Never triggers a React re-render.
 */
export function useCursorTrack() {
  const raw = useRef<CursorVec>({ x: 0, y: 0 });
  const lerped = useRef<CursorVec>({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      raw.current.x = (e.clientX / window.innerWidth - 0.5) * 2;
      raw.current.y = (e.clientY / window.innerHeight - 0.5) * 2;
    };
    window.addEventListener("mousemove", onMove, { passive: true });
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const tick = (factor = 0.05) => {
    lerped.current.x += (raw.current.x - lerped.current.x) * factor;
    lerped.current.y += (raw.current.y - lerped.current.y) * factor;
    return lerped.current;
  };

  return { raw, lerped, tick };
}
