export default function LiveBadge({ label = "LIVE" }: { label?: string }) {
  return (
    <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase text-data-green">
      <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-data-green" />
      {label}
    </span>
  );
}
