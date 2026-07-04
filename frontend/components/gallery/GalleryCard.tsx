"use client";

import { type ReactNode, type MouseEvent } from "react";
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion";
import { cn } from "@/lib/utils";

const TILT_RANGE_DEG = 5;

export default function GalleryCard({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const px = useMotionValue(0);
  const py = useMotionValue(0);

  const rotateX = useSpring(useTransform(py, [-0.5, 0.5], [TILT_RANGE_DEG, -TILT_RANGE_DEG]), {
    stiffness: 220,
    damping: 22,
  });
  const rotateY = useSpring(useTransform(px, [-0.5, 0.5], [-TILT_RANGE_DEG, TILT_RANGE_DEG]), {
    stiffness: 220,
    damping: 22,
  });
  const scale = useSpring(1, { stiffness: 220, damping: 20 });

  function onMouseMove(e: MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    px.set((e.clientX - rect.left) / rect.width - 0.5);
    py.set((e.clientY - rect.top) / rect.height - 0.5);
  }

  function onMouseEnter() {
    scale.set(1.04);
  }

  function onMouseLeave() {
    scale.set(1);
    px.set(0);
    py.set(0);
  }

  return (
    <motion.div
      onMouseMove={onMouseMove}
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{ rotateX, rotateY, scale, transformPerspective: 900 }}
      className={cn(
        "relative w-full overflow-hidden rounded-lg border border-border-dim bg-carbon-3",
        className
      )}
    >
      {children}
    </motion.div>
  );
}
