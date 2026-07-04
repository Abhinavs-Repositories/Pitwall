import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        carbon: {
          DEFAULT: "var(--carbon)",
          2: "var(--carbon-2)",
          3: "var(--carbon-3)",
          4: "var(--carbon-4)",
        },
        "f1-red": {
          DEFAULT: "var(--f1-red)",
          dim: "var(--f1-red-dim)",
          glow: "var(--f1-red-glow)",
        },
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          muted: "var(--text-muted)",
        },
        data: {
          green: "var(--data-green)",
          amber: "var(--data-amber)",
          blue: "var(--data-blue)",
        },
        border: {
          dim: "var(--border-dim)",
          bright: "var(--border-bright)",
          accent: "var(--border-accent)",
        },
        team: {
          redbull: "var(--team-redbull)",
          ferrari: "var(--team-ferrari)",
          mercedes: "var(--team-mercedes)",
          mclaren: "var(--team-mclaren)",
          alpine: "var(--team-alpine)",
          aston: "var(--team-aston)",
          williams: "var(--team-williams)",
          haas: "var(--team-haas)",
          sauber: "var(--team-sauber)",
          rb: "var(--team-rb)",
        },
        compound: {
          soft: "var(--compound-soft)",
          medium: "var(--compound-medium)",
          hard: "var(--compound-hard)",
          inter: "var(--compound-inter)",
          wet: "var(--compound-wet)",
        },
      },
      fontFamily: {
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
        body: ["var(--font-body)"],
      },
      borderRadius: {
        sm: "var(--radius-sm)",
        md: "var(--radius-md)",
        lg: "var(--radius-lg)",
      },
    },
  },
  plugins: [],
};
export default config;
