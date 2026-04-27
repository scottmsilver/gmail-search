// Allow `await import("…/something.css")` for side-effect CSS imports
// that we load dynamically (so SSR doesn't try to evaluate them).
// x-data-spreadsheet ships its stylesheet at this path.
declare module "*.css";
declare module "x-data-spreadsheet/dist/xspreadsheet.css";
