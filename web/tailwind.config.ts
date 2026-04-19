import type { Config } from "tailwindcss";
import typography from "@tailwindcss/typography";
import animate from "tailwindcss-animate";

const config: Config = {
  darkMode: ["class", '[data-theme="dark"]', '[data-theme="slate"]'],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // shadcn-style semantic tokens mapped onto the repo's existing
        // --bg-*/--fg-* theme variables so every theme (light/dark/
        // sepia/slate) flows through unchanged.
        background: "var(--bg-page)",
        foreground: "var(--fg-primary)",
        card: {
          DEFAULT: "var(--bg-surface)",
          foreground: "var(--fg-primary)",
        },
        popover: {
          DEFAULT: "var(--bg-surface)",
          foreground: "var(--fg-primary)",
        },
        primary: {
          DEFAULT: "var(--accent)",
          foreground: "var(--accent-fg)",
        },
        secondary: {
          DEFAULT: "var(--bg-raised)",
          foreground: "var(--fg-primary)",
        },
        muted: {
          DEFAULT: "var(--bg-raised)",
          foreground: "var(--fg-muted)",
        },
        accent: {
          DEFAULT: "var(--bg-raised)",
          foreground: "var(--fg-primary)",
        },
        destructive: {
          DEFAULT: "hsl(0 72% 51%)",
          foreground: "hsl(0 0% 100%)",
        },
        border: "var(--border-subtle)",
        input: "var(--border-default)",
        ring: "var(--accent)",
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [typography, animate],
};

export default config;
