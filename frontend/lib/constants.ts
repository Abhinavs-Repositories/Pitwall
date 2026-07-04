import type { TireCompound } from "./types";

export const COMPOUND_COLOR: Record<TireCompound, string> = {
  SOFT: "var(--compound-soft)",
  MEDIUM: "var(--compound-medium)",
  HARD: "var(--compound-hard)",
  INTERMEDIATE: "var(--compound-inter)",
  WET: "var(--compound-wet)",
  UNKNOWN: "var(--text-muted)",
};

export const COMPOUND_LETTER: Record<TireCompound, string> = {
  SOFT: "S",
  MEDIUM: "M",
  HARD: "H",
  INTERMEDIATE: "I",
  WET: "W",
  UNKNOWN: "?",
};

// Best-effort team → brand colour lookup (OpenF1 team names vary slightly
// season to season, hence the substring matching in teamColor()).
export const TEAM_COLOR: Record<string, string> = {
  "red bull": "var(--team-redbull)",
  ferrari: "var(--team-ferrari)",
  mercedes: "var(--team-mercedes)",
  mclaren: "var(--team-mclaren)",
  alpine: "var(--team-alpine)",
  "aston martin": "var(--team-aston)",
  williams: "var(--team-williams)",
  haas: "var(--team-haas)",
  sauber: "var(--team-audi-silver)",
  audi: "var(--team-audi)",
  "racing bulls": "var(--team-rb)",
  rb: "var(--team-rb)",
  alphatauri: "var(--team-rb)",
};

export function teamColor(team: string): string {
  const key = team.toLowerCase();
  for (const [needle, color] of Object.entries(TEAM_COLOR)) {
    if (key.includes(needle)) return color;
  }
  return "var(--text-muted)";
}

export function formatGap(gap: number | null): string {
  if (gap === null || gap === undefined) return "—";
  if (gap === 0) return "LEADER";
  return `+${gap.toFixed(3)}s`;
}

export function formatLapTime(seconds: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  const mins = Math.floor(seconds / 60);
  const secs = (seconds % 60).toFixed(3).padStart(6, "0");
  return `${mins}:${secs}`;
}
