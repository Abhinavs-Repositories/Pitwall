"use client";

import { createContext, useContext, useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Lenis from "lenis";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

const LenisContext = createContext<Lenis | null>(null);

export function useLenis() {
  return useContext(LenisContext);
}

// Lenis's smooth momentum scroll is a landing-page aesthetic choice — it
// hijacks wheel events globally, which breaks native scrolling inside
// nested panels on functional "app" pages like /strategy. Only run it on
// the cinematic marketing route.
const LENIS_ROUTES = ["/"];

export function LenisProvider({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [instance, setInstance] = useState<Lenis | null>(null);
  const enabled = LENIS_ROUTES.includes(pathname);

  useEffect(() => {
    if (!enabled) {
      setInstance(null);
      return;
    }

    const lenis = new Lenis({
      duration: 1.4,
      easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smoothWheel: true,
    });
    setInstance(lenis);

    lenis.on("scroll", ScrollTrigger.update);

    const raf = (time: number) => {
      lenis.raf(time * 1000);
    };
    gsap.ticker.add(raf);
    gsap.ticker.lagSmoothing(0);

    return () => {
      gsap.ticker.remove(raf);
      lenis.destroy();
    };
  }, [enabled]);

  return <LenisContext.Provider value={instance}>{children}</LenisContext.Provider>;
}
