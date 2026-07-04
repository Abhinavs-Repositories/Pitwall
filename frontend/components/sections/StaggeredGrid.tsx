"use client";

import { motion } from "framer-motion";

// Real circuit outlines (public domain, openclipart.org), recolored to
// match our F1-red accent — not generative approximations.
const CIRCUITS = [
  { name: "MONACO", file: "monaco.svg" },
  { name: "SILVERSTONE", file: "silverstone.svg" },
  { name: "MONZA", file: "monza.svg" },
  { name: "SUZUKA", file: "suzuka.svg" },
  { name: "SPA-FRANCORCHAMPS", file: "spa.svg" },
  { name: "INTERLAGOS", file: "interlagos.svg" },
  { name: "MARINA BAY", file: "marina-bay.svg" },
  { name: "COTA", file: "cota.svg" },
  { name: "MELBOURNE", file: "melbourne.svg" },
];

const gridVariants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.08, delayChildren: 0.1 } },
};

const itemVariants = {
  hidden: { opacity: 0, scale: 0.9, y: 20 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export default function StaggeredGrid() {
  return (
    <section className="w-full bg-carbon px-6 py-24 md:px-16">
      <motion.div
        className="grid grid-cols-1 gap-4 sm:grid-cols-3"
        variants={gridVariants}
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
      >
        {CIRCUITS.map((c) => (
          <motion.div
            key={c.name}
            variants={itemVariants}
            className="rounded-lg border border-border-dim bg-carbon-3 p-5"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={`/circuits/${c.file}`}
              alt={`${c.name} circuit outline`}
              className="h-20 w-full object-contain"
            />
            <p className="mt-3 font-mono text-xs uppercase tracking-widest text-text-secondary">
              {c.name}
            </p>
          </motion.div>
        ))}
      </motion.div>
    </section>
  );
}
