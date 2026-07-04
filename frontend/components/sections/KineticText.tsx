"use client";

import { useMemo, useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";

function seededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

// Deterministic wavy contour lines standing in for a topographic map —
// no map data needed, just enough visual texture to break up the flat bg.
function useTopoPaths() {
  return useMemo(() => {
    const rand = seededRandom(11);
    return Array.from({ length: 9 }, (_, i) => {
      const y = 60 + i * 100;
      const amp = 10 + rand() * 26;
      const freq = 0.006 + rand() * 0.004;
      let d = `M0,${y.toFixed(1)}`;
      for (let x = 0; x <= 1440; x += 30) {
        d += ` L${x},${(y + Math.sin(x * freq) * amp).toFixed(1)}`;
      }
      return d;
    });
  }, []);
}

export default function KineticText() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const topoRef = useRef<SVGSVGElement>(null);
  const strategyRef = useRef<HTMLSpanElement>(null);
  const isRef = useRef<HTMLSpanElement>(null);
  const everythingRef = useRef<HTMLSpanElement>(null);
  const topoPaths = useTopoPaths();

  useGSAP(
    () => {
      const trigger = {
        trigger: sectionRef.current,
        start: "top 70%",
        toggleActions: "play none none reverse",
      };

      // Each line gets its own vector and personality — deliberately not a
      // mechanical left/right/left alternation.
      gsap.fromTo(
        strategyRef.current,
        { x: -420, opacity: 0 },
        { x: 0, opacity: 1, duration: 1.0, ease: "power3.out", scrollTrigger: trigger }
      );
      gsap.fromTo(
        isRef.current,
        { y: 90, opacity: 0, scale: 0.8 },
        {
          y: 0,
          opacity: 1,
          scale: 1,
          duration: 0.8,
          delay: 0.18,
          ease: "back.out(1.6)",
          scrollTrigger: trigger,
        }
      );
      gsap.fromTo(
        everythingRef.current,
        { x: 420, opacity: 0 },
        { x: 0, opacity: 1, duration: 1.0, delay: 0.34, ease: "power3.out", scrollTrigger: trigger }
      );

      gsap.fromTo(
        topoRef.current,
        { opacity: 0 },
        { opacity: 0.04, duration: 1.4, scrollTrigger: trigger }
      );

      gsap.fromTo(
        ".kinetic-subcopy",
        { opacity: 0, y: 20 },
        { opacity: 1, y: 0, duration: 0.8, delay: 0.6, scrollTrigger: trigger }
      );
    },
    { scope: sectionRef }
  );

  return (
    <section
      ref={sectionRef}
      className="kinetic-section relative flex min-h-screen w-full flex-col items-center justify-center gap-8 overflow-hidden bg-carbon-2 py-32"
    >
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(45% 40% at 15% 20%, rgba(0,144,255,0.09), transparent 70%), " +
            "radial-gradient(45% 40% at 85% 80%, rgba(255,128,0,0.07), transparent 70%)",
        }}
      />
      <svg
        ref={topoRef}
        className="pointer-events-none absolute inset-0 h-full w-full opacity-0"
        preserveAspectRatio="none"
        viewBox="0 0 1440 900"
      >
        {topoPaths.map((d, i) => (
          <path key={i} d={d} stroke="#ffffff" strokeWidth={1} fill="none" />
        ))}
      </svg>

      <div className="relative z-10 flex flex-col items-center gap-1 text-center">
        <div className="overflow-hidden">
          <span
            ref={strategyRef}
            className="inline-block font-display font-bold uppercase leading-[0.95] text-text-primary"
            style={{ fontSize: "clamp(4rem, 12vw, 10rem)" }}
          >
            STRATEGY
          </span>
        </div>
        <div className="overflow-hidden">
          <span
            ref={isRef}
            className="inline-block font-display font-bold uppercase leading-[0.95] text-text-primary"
            style={{ fontSize: "clamp(4rem, 12vw, 10rem)" }}
          >
            IS
          </span>
        </div>
        <div className="overflow-hidden">
          <span
            ref={everythingRef}
            className="inline-block font-display font-bold uppercase leading-[0.95] text-text-primary"
            style={{ fontSize: "clamp(4rem, 12vw, 10rem)" }}
          >
            EVERYTHING.
          </span>
        </div>
      </div>

      <p className="kinetic-subcopy relative z-10 font-mono text-sm tracking-wide text-text-secondary">
        // 7 agents. live telemetry. pit-wall grade answers.
      </p>
    </section>
  );
}
