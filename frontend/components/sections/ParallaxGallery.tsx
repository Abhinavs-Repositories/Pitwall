"use client";

import { useRef, type ComponentType } from "react";
import gsap from "gsap";
import { useGSAP } from "@gsap/react";
import GalleryCard from "@/components/gallery/GalleryCard";
import TireCompoundPanel from "@/components/gallery/TireCompoundPanel";
import SectorDeltaPanel from "@/components/gallery/SectorDeltaPanel";
import CircuitMapPanel from "@/components/gallery/CircuitMapPanel";
import TireDegPanel from "@/components/gallery/TireDegPanel";
import UndercutPanel from "@/components/gallery/UndercutPanel";
import TelemetryWavePanel from "@/components/gallery/TelemetryWavePanel";
import WeatherGaugePanel from "@/components/gallery/WeatherGaugePanel";

interface CardSpec {
  Panel: ComponentType;
  offset: number;
  height: string;
}

interface ColumnSpec {
  factor: number;
  cards: CardSpec[];
}

// Three hand-balanced columns rather than CSS auto-flow masonry — auto-flow
// left one column with three cards and two with only two, leaving a big
// trailing gap. Explicit placement keeps the collage irregular without
// the accidental whitespace.
//
// Parallax factor lives on the COLUMN, not the card: cards stacked in the
// same column share one scroll speed. Giving neighbours very different
// factors (tried 0.7 vs 1.4 initially) makes the faster one drift up and
// overlap the slower one above it as the page scrolls — different speeds
// only work across columns, which don't share vertical space.
const COLUMNS: ColumnSpec[] = [
  {
    factor: 0.75,
    cards: [
      { Panel: SectorDeltaPanel, offset: 34, height: "h-56" },
      { Panel: UndercutPanel, offset: -14, height: "h-56" },
      { Panel: TelemetryWavePanel, offset: 20, height: "h-60" },
    ],
  },
  {
    factor: 1.2,
    cards: [
      { Panel: TireCompoundPanel, offset: -18, height: "h-72" },
      { Panel: CircuitMapPanel, offset: 22, height: "h-80" },
    ],
  },
  {
    factor: 0.95,
    cards: [
      { Panel: TireDegPanel, offset: -10, height: "h-64" },
      { Panel: WeatherGaugePanel, offset: 16, height: "h-72" },
    ],
  },
];

export default function ParallaxGallery() {
  const sectionRef = useRef<HTMLDivElement>(null);

  useGSAP(
    () => {
      const columns = gsap.utils.toArray<HTMLElement>(".parallax-column");
      columns.forEach((column) => {
        const factor = parseFloat(column.dataset.parallax ?? "1");
        gsap.to(column, {
          yPercent: -20 * factor,
          ease: "none",
          scrollTrigger: {
            trigger: column,
            start: "top bottom",
            end: "bottom top",
            scrub: 1,
          },
        });
      });
    },
    { scope: sectionRef }
  );

  return (
    <section ref={sectionRef} className="relative w-full bg-carbon-2 px-6 py-32 md:px-16">
      <div className="mb-16 max-w-2xl">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-text-muted">
          // telemetry deck
        </p>
        <h2 className="mt-3 font-display text-4xl font-bold uppercase text-text-primary md:text-5xl">
          Every number pit wall sees
        </h2>
      </div>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {COLUMNS.map(({ factor, cards }, colIndex) => (
          <div
            key={colIndex}
            className="parallax-column flex flex-col gap-6"
            data-parallax={factor}
          >
            {cards.map(({ Panel, offset, height }, i) => (
              <div key={i} style={{ transform: `translateY(${offset}px)` }}>
                <GalleryCard className={height}>
                  <Panel />
                </GalleryCard>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}
