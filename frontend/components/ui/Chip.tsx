import { cn } from "@/lib/utils";

export default function Chip({
  children,
  onClick,
  className,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-full border border-border-dim bg-carbon-3 px-3 py-1.5 font-mono text-xs text-text-secondary transition-colors hover:border-border-accent hover:text-text-primary",
        className
      )}
    >
      {children}
    </button>
  );
}
