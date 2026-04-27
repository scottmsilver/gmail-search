// Empty stub used by next.config.ts to redirect non-CSS-pipeline
// stylesheet imports (currently x-data-spreadsheet's index.less) to
// nothing. We import the precompiled dist CSS directly in the
// component, so the source-side .less requires can safely resolve to
// an empty module instead of failing at runtime.
module.exports = {};
