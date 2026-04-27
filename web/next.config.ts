import path from "node:path";

import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Hide the little "N" build-status indicator in the bottom-left corner
  // during `next dev` — it overlaps the composer input.
  devIndicators: false,
  webpack: (webpackConfig, { webpack }) => {
    // x-data-spreadsheet's `main` points at `src/index.js`, which imports
    // `index.less`. Next.js doesn't ship a less-loader, and we already
    // import the precompiled `dist/xspreadsheet.css` separately. Replace
    // any .less request with an empty stub module so the import succeeds
    // (returns `{}`) without ever needing to actually parse less. We use
    // NormalModuleReplacementPlugin instead of IgnorePlugin because
    // IgnorePlugin makes the import THROW at runtime — replacement makes
    // it resolve to a real (empty) module.
    //
    // We can't add a `module.rules` entry for .less either: Next.js
    // detects that as "custom CSS config" and disables its own CSS
    // pipeline (breaking globals.css with @tailwind).
    const emptyStub = path.resolve(__dirname, "lib/empty-module.js");
    webpackConfig.plugins.push(
      new webpack.NormalModuleReplacementPlugin(/\.less$/, emptyStub),
    );
    return webpackConfig;
  },
};

export default config;
