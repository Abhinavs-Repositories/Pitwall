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

export interface TeamTheme {
  name: string;
  primary: string;
  secondary: string;
  logo: string | null;
}

// Official liveries per season, sourced from Fast-F1's constants.json
// (github.com/theOehrly/Fast-F1) rather than guessed — colors drift from
// year to year so this is the actual per-season reference, not a single
// "current" palette. Secondary accents are hand-picked from each team's
// well-known livery detailing (not in Fast-F1, which only tracks one
// color per team). Historical entities (Alfa Romeo -> Sauber, AlphaTauri
// -> RB) are folded into their successor's bucket since our race data
// spans 2023-2025 and both names appear.
const THEMES: Record<string, TeamTheme> = {
  "red bull": {
    name: "Red Bull Racing",
    primary: "#3671c6",
    secondary: "#ffc906",
    logo: "/team-logos/redbull.png",
  },
  ferrari: {
    name: "Scuderia Ferrari",
    primary: "#e80020",
    secondary: "#fff200",
    logo: "/team-logos/ferrari.png",
  },
  mercedes: {
    name: "Mercedes-AMG Petronas",
    primary: "#27f4d2",
    secondary: "#00a19a",
    logo: "/team-logos/mercedes.png",
  },
  mclaren: {
    name: "McLaren",
    primary: "#ff8000",
    secondary: "#00a3e0",
    logo: "/team-logos/mclaren.png",
  },
  alpine: {
    name: "BWT Alpine",
    primary: "#0093cc",
    secondary: "#ff87bc",
    logo: "/team-logos/alpine.png",
  },
  "aston martin": {
    name: "Aston Martin Aramco",
    primary: "#229971",
    secondary: "#cedc00",
    logo: "/team-logos/astonmartin.png",
  },
  williams: {
    name: "Williams Racing",
    primary: "#64c4ff",
    secondary: "#041e42",
    logo: "/team-logos/williams.png",
  },
  haas: {
    name: "Haas F1 Team",
    primary: "#b6babd",
    secondary: "#e6002b",
    logo: "/team-logos/haas.png",
  },
  sauber: {
    name: "Kick Sauber",
    primary: "#52e252",
    secondary: "#111111",
    logo: "/team-logos/sauber.png",
  },
  "alfa romeo": {
    name: "Alfa Romeo",
    primary: "#c92d4b",
    secondary: "#111111",
    logo: "/team-logos/sauber.png",
  },
  "racing bulls": {
    name: "Racing Bulls",
    primary: "#6692ff",
    secondary: "#ffffff",
    logo: "/team-logos/rb.png",
  },
  alphatauri: {
    name: "AlphaTauri",
    primary: "#5e8faa",
    secondary: "#ffffff",
    logo: "/team-logos/rb.png",
  },
  rb: {
    name: "RB",
    primary: "#6692ff",
    secondary: "#ffffff",
    logo: "/team-logos/rb.png",
  },
};

export const DEFAULT_THEME: TeamTheme = {
  name: "Pitwall",
  primary: "#e8002d",
  secondary: "#ff8000",
  logo: null,
};

export function getTeamTheme(team: string | undefined): TeamTheme {
  if (!team) return DEFAULT_THEME;
  const key = team.toLowerCase();
  for (const [needle, theme] of Object.entries(THEMES)) {
    if (key.includes(needle)) return theme;
  }
  return DEFAULT_THEME;
}

export function teamColor(team: string): string {
  return getTeamTheme(team).primary;
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
