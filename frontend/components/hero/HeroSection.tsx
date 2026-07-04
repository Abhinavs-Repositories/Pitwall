"use client";

import { useRef } from "react";
import dynamic from "next/dynamic";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import HeroText from "./HeroText";
import SpeedStreakCanvas from "./SpeedStreakCanvas";

const CircuitCanvas = dynamic(() => import("./CircuitCanvas"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 bg-carbon" />,
});

export default function HeroSection() {
  const root = useRef<HTMLDivElement>(null);
  const bgRef = useRef<HTMLDivElement>(null);
  const visualRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const underlineRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const tl = gsap.timeline({ defaults: { ease: "power3.out" } });
      tl.fromTo(bgRef.current, { opacity: 0 }, { opacity: 1, duration: 0.8 }, 0)
        .fromTo(
          visualRef.current,
          { opacity: 0, xPercent: -4 },
          { opacity: 1, xPercent: 0, duration: 1 },
          0.3
        )
        .fromTo(
          canvasRef.current,
          { opacity: 0, xPercent: 4 },
          { opacity: 1, xPercent: 0, duration: 1 },
          0.3
        )
        .fromTo(textRef.current, { opacity: 0, y: 40 }, { opacity: 1, y: 0, duration: 0.9 }, 0.6);

      if (underlineRef.current) {
        gsap.fromTo(
          underlineRef.current,
          { scaleX: 0 },
          { scaleX: 1, duration: 1.2, ease: "power2.out", delay: 0.9 }
        );
      }
    },
    { scope: root }
  );

  return (
    <section ref={root} className="relative h-screen w-full overflow-hidden bg-carbon">
      <div ref={bgRef} className="absolute inset-0">
        <div className="grain-overlay" />
      </div>

      <div className="relative grid h-full w-full grid-cols-1 md:grid-cols-2">
        <div ref={visualRef} className="relative h-full w-full">
          <SpeedStreakCanvas />
        </div>
        <div ref={canvasRef} className="relative h-full w-full">
          <CircuitCanvas />
        </div>
      </div>

      <div
        ref={textRef}
        className="pointer-events-none absolute inset-0 flex items-center justify-center"
      >
        <HeroText ref={underlineRef} />
      </div>
    </section>
  );
}
