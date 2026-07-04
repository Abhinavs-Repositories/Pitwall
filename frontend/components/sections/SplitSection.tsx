"use client";

import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import ConsolePreview from "@/components/strategy/ConsolePreview";
import OnTrackVisual from "@/components/sections/OnTrackVisual";

export default function SplitSection() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const trigger = {
        trigger: sectionRef.current,
        start: "top 65%",
        toggleActions: "play none none reverse",
      };
      gsap.fromTo(
        leftRef.current,
        { xPercent: -60, opacity: 0 },
        { xPercent: 0, opacity: 1, duration: 1.1, ease: "power3.out", scrollTrigger: trigger }
      );
      gsap.fromTo(
        rightRef.current,
        { xPercent: 60, opacity: 0 },
        { xPercent: 0, opacity: 1, duration: 1.1, ease: "power3.out", scrollTrigger: trigger }
      );
    },
    { scope: sectionRef }
  );

  return (
    <section ref={sectionRef} className="relative w-full overflow-hidden bg-carbon">
      <div className="grid min-h-[85vh] grid-cols-1 md:grid-cols-2">
        <div ref={leftRef} className="flex flex-col justify-center gap-6 p-10 md:p-16">
          <p className="font-mono text-xs uppercase tracking-[0.3em] text-text-muted">
            // on pit wall
          </p>
          <h3 className="font-display text-3xl font-bold uppercase text-text-primary md:text-4xl">
            Where the call gets made
          </h3>
          <ConsolePreview />
        </div>

        <div ref={rightRef} className="relative min-h-[50vh] md:min-h-0">
          <OnTrackVisual />
          <div className="absolute bottom-10 left-10 right-10">
            <p className="font-mono text-xs uppercase tracking-[0.3em] text-text-muted">
              // on track
            </p>
            <h3 className="font-display text-3xl font-bold uppercase text-text-primary md:text-4xl">
              Where it plays out
            </h3>
          </div>
        </div>
      </div>
    </section>
  );
}
