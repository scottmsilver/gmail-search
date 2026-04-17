import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Hide the little "N" build-status indicator in the bottom-left corner
  // during `next dev` — it overlaps the composer input.
  devIndicators: false,
};

export default config;
