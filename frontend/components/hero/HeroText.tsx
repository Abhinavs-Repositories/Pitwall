"use client";

import { forwardRef } from "react";

const HeroText = forwardRef<HTMLDivElement>(function HeroText(_, underlineRef) {
  return (
    <div className="flex flex-col items-center gap-4 text-center">
      <h1 className="font-display font-bold uppercase leading-none tracking-tight text-text-primary text-[14vw] select-none">
        PITWALL
      </h1>
      <div className="h-[2px] w-[46vw] max-w-[560px] overflow-hidden">
        <div
          ref={underlineRef}
          className="h-full w-full origin-left"
          style={{ background: "linear-gradient(90deg, #e8002d, #ff8000)" }}
        />
      </div>
      <p className="font-mono text-xs tracking-[0.3em] text-text-muted uppercase mt-6 animate-pulse">
        // scroll to engage
      </p>
    </div>
  );
});

export default HeroText;
