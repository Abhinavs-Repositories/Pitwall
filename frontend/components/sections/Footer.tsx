import Link from "next/link";

export default function Footer() {
  return (
    <footer className="flex flex-col items-center gap-6 bg-carbon py-24">
      <p className="font-display text-2xl font-bold uppercase tracking-wide text-text-primary">
        PITWALL
      </p>
      <Link
        href="/strategy"
        className="rounded-full border border-f1-red px-6 py-3 font-mono text-sm uppercase tracking-widest text-f1-red transition-colors hover:bg-f1-red hover:text-text-primary"
      >
        Launch Strategy Console →
      </Link>
    </footer>
  );
}
