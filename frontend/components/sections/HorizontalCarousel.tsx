"use client";

import { useRef } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import { CALENDAR_2026 } from "@/lib/calendar2026";

export default function HorizontalCarousel() {
  const outerRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const outer = outerRef.current;
      const inner = innerRef.current;
      if (!outer || !inner) return;

      const getMaxScroll = () => inner.scrollWidth - outer.clientWidth;

      gsap.to(inner, {
        x: () => -getMaxScroll(),
        ease: "none",
        scrollTrigger: {
          trigger: outer,
          start: "top top",
          end: () => `+=${getMaxScroll()}`,
          pin: true,
          scrub: 1.5,
          anticipatePin: 1,
          invalidateOnRefresh: true,
        },
      });
    },
    { scope: outerRef }
  );

  return (
    <section ref={outerRef} className="relative h-screen w-full overflow-hidden bg-carbon-2">
      <div className="pointer-events-none absolute left-6 top-10 z-10 md:left-16">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-text-muted">
          // 2026 calendar
        </p>
        <h2 className="mt-2 font-display text-3xl font-bold uppercase text-text-primary md:text-4xl">
          24 races. One brain.
        </h2>
      </div>

      <div ref={innerRef} className="flex h-full w-max items-center gap-6 pl-6 pr-24 pt-28 md:pl-16">
        {CALENDAR_2026.map((race) => (
          <div
            key={race.round}
            className="flex h-[58vh] w-[280px] flex-shrink-0 flex-col justify-between rounded-lg border border-border-dim bg-carbon-3 p-6"
          >
            <div className="flex items-center justify-between font-mono text-xs text-text-muted">
              <span>RD {String(race.round).padStart(2, "0")}</span>
              <span>{race.dateRange}</span>
            </div>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/calendar/${race.shapeFile}`}
              alt={`${race.location} circuit outline`}
              className="h-36 w-full object-contain"
            />
            <div>
              <p className="font-display text-xl font-bold uppercase text-text-primary">
                {race.location}
              </p>
              <p className="font-mono text-xs text-text-secondary">{race.name}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
