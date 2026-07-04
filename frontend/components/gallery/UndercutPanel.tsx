export default function UndercutPanel() {
  return (
    <div className="flex h-full flex-col justify-between p-6">
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-text-muted">
        // undercut window
      </p>
      <div className="relative flex items-center py-10">
        <div className="h-px w-full bg-border-bright" />
        <div className="absolute left-[16%] -translate-x-1/2 flex flex-col items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-team-redbull shadow-[0_0_10px_rgba(54,113,198,0.7)]" />
          <span className="font-mono text-[10px] text-text-secondary">CAR A</span>
        </div>
        <div className="absolute left-[58%] -translate-x-1/2 flex flex-col items-center gap-2">
          <div className="h-3 w-3 rounded-full bg-f1-red shadow-[0_0_10px_rgba(232,0,32,0.7)]" />
          <span className="font-mono text-[10px] text-text-secondary">CAR B</span>
        </div>
      </div>
      <div className="flex items-baseline justify-between">
        <p className="font-display text-xl font-bold text-text-primary">Undercut Viable</p>
        <span className="font-mono text-xs text-data-green">GAP 1.8s</span>
      </div>
    </div>
  );
}
