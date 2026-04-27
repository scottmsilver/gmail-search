"use client";

import { useEffect, useRef, useState } from "react";

// XLSX preview, rendered with x-data-spreadsheet (a.k.a. x-spreadsheet)
// for an Excel-like look: gridlines, cell borders, sheet tabs at the
// bottom, row/column headers, scroll. We parse the bytes with SheetJS
// (`xlsx`) — its `sheet_to_json({header:1})` handles every xlsx
// variant openpyxl, Excel, or LibreOffice produces — then convert
// each sheet's Cell[][] into the {rows:{n:{cells:{m:{text}}}}} shape
// x-spreadsheet expects.
//
// Read-only by design — `mode: "read"` plus disabling the toolbar +
// context menu so the preview can't accidentally mutate the workbook.

type Cell = string | number | boolean | Date | null;
type SheetRows = Cell[][];
type ParsedSheet = { sheet: string; data: SheetRows };

const ROW_CAP = 200;

// ── Public component ──────────────────────────────────────────────

export const XlsxTable = ({ url }: { url: string }) => {
  const { sheets, err } = useXlsxSheets(url);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [mountErr, setMountErr] = useState<string | null>(null);

  useMountSpreadsheet(containerRef, sheets, setMountErr);

  if (err) return <div className="p-6 text-sm text-red-600">Error: {err}</div>;
  if (sheets === null)
    return <div className="p-6 text-sm text-neutral-500">Loading XLSX…</div>;
  if (sheets.length === 0)
    return <div className="p-6 text-sm text-neutral-500">Empty workbook.</div>;
  if (mountErr) return <div className="p-6 text-sm text-red-600">Error: {mountErr}</div>;

  const truncated = sheets.some((s) => s.data.length - 1 > ROW_CAP);

  return (
    <div className="flex h-full flex-col">
      <div ref={containerRef} className="x-spreadsheet-host flex-1 overflow-hidden" />
      {truncated && (
        <div className="border-t bg-neutral-50 px-3 py-2 text-[11px] text-neutral-500">
          Showing first {ROW_CAP} rows per sheet. Download for full XLSX.
        </div>
      )}
    </div>
  );
};

// ── Fetch + parse ─────────────────────────────────────────────────

const useXlsxSheets = (
  url: string,
): { sheets: ParsedSheet[] | null; err: string | null } => {
  const [sheets, setSheets] = useState<ParsedSheet[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    setSheets(null);
    setErr(null);
    void fetchAndParse(url)
      .then((parsed) => {
        if (cancelled) return;
        if (parsed.length === 0) setErr("Empty workbook.");
        else setSheets(parsed);
      })
      .catch((e) => {
        if (!cancelled) setErr(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [url]);
  return { sheets, err };
};

const fetchAndParse = async (url: string): Promise<ParsedSheet[]> => {
  const buf = await fetchAsArrayBuffer(url);
  return parseWorkbook(buf);
};

const fetchAsArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.arrayBuffer();
};

const parseWorkbook = async (buf: ArrayBuffer): Promise<ParsedSheet[]> => {
  // Dynamic import keeps the parser out of the SSR bundle.
  // SheetJS handles every xlsx variant (inline strings, shared strings,
  // formula cells, dates) — read-excel-file threw on empty inlineStr
  // cells produced by openpyxl.
  const XLSX = await import("xlsx");
  const workbook = XLSX.read(new Uint8Array(buf), { type: "array" });
  return workbook.SheetNames.map((name) => {
    const sheet = workbook.Sheets[name];
    const aoa = XLSX.utils.sheet_to_json(sheet, {
      header: 1,
      defval: "",
      blankrows: false,
    }) as unknown as SheetRows;
    return { sheet: name, data: aoa };
  });
};

// ── Cell -> x-spreadsheet text ────────────────────────────────────

const formatCell = (v: Cell): string => {
  if (v === null || v === undefined) return "";
  if (isDate(v)) return v.toISOString().slice(0, 10);
  if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
  return String(v);
};

const isDate = (v: unknown): v is Date =>
  typeof v === "object" && v !== null && Object.prototype.toString.call(v) === "[object Date]";

// ── Cell[][]  →  x-spreadsheet sheet data ─────────────────────────
//
// x-spreadsheet's loadData accepts either a single sheet object or an
// array of them. Each sheet has shape:
//
//   { name, rows: { 0: { cells: { 0: {text}, 1: {text}, ... } }, ... } }
//
// We slice to ROW_CAP rows and stamp the header row (row 0) with bold
// styling so the preview reads like a workbook header, not data.

const HEADER_STYLE = {
  font: { bold: true },
  bgcolor: "#f3f4f6",
};

type XSheet = {
  name: string;
  styles: Array<typeof HEADER_STYLE>;
  rows: { len: number; [k: number]: { cells: { [k: number]: { text: string; style?: number } } } };
};

const sheetToXSpreadsheetData = (sheet: ParsedSheet): XSheet => {
  const truncated = sheet.data.slice(0, ROW_CAP);
  const rows: XSheet["rows"] = { len: Math.max(truncated.length, 1) };
  truncated.forEach((row, ri) => {
    const cells: { [k: number]: { text: string; style?: number } } = {};
    row.forEach((cell, ci) => {
      const text = formatCell(cell);
      if (text === "" && ri !== 0) return;
      cells[ci] = ri === 0 ? { text, style: 0 } : { text };
    });
    rows[ri] = { cells };
  });
  return { name: sheet.sheet || "Sheet", styles: [HEADER_STYLE], rows };
};

// ── Mounting x-spreadsheet ────────────────────────────────────────

const useMountSpreadsheet = (
  containerRef: React.RefObject<HTMLDivElement | null>,
  sheets: ParsedSheet[] | null,
  setErr: (e: string) => void,
) => {
  useEffect(() => {
    if (!sheets || !containerRef.current) return;
    const el = containerRef.current;
    let disposed = false;
    let resizeObs: ResizeObserver | null = null;

    void mountSpreadsheet(el, sheets, () => disposed)
      .then((cleanup) => {
        if (disposed) {
          cleanup?.();
          return;
        }
        resizeObs = cleanup?.resizeObs ?? null;
      })
      .catch((e) => {
        if (!disposed) setErr(e instanceof Error ? e.message : String(e));
      });

    return () => {
      disposed = true;
      resizeObs?.disconnect();
      el.innerHTML = "";
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sheets]);
};

type MountCleanup = (() => void) & { resizeObs?: ResizeObserver };

const mountSpreadsheet = async (
  el: HTMLDivElement,
  sheets: ParsedSheet[],
  isDisposed: () => boolean,
): Promise<MountCleanup | undefined> => {
  // Dynamic import — both the JS module and its CSS — so SSR doesn't
  // try to evaluate browser-only code or parse a bare CSS import.
  await import("x-data-spreadsheet/dist/xspreadsheet.css");
  const mod = await import("x-data-spreadsheet");
  if (isDisposed()) return;
  const Spreadsheet = mod.default;

  const data = sheets.map(sheetToXSpreadsheetData);
  const ss = new Spreadsheet(el, {
    mode: "read",
    showToolbar: false,
    showGrid: true,
    showContextmenu: false,
    showBottomBar: true,
    view: {
      width: () => el.clientWidth || 600,
      height: () => el.clientHeight || 400,
    },
  });
  ss.loadData(data as unknown as Record<string, unknown>);

  // x-spreadsheet measures its canvas at construction time; if the
  // container hadn't laid out yet the grid renders at 0×0. Kick a
  // re-render after the next frame, then re-render on resize.
  const reRender = () => {
    const anyss = ss as unknown as { reRender?: () => void };
    anyss.reRender?.();
  };
  setTimeout(reRender, 0);
  const resizeObs = new ResizeObserver(() => reRender());
  resizeObs.observe(el);

  const cleanup = (() => {
    resizeObs.disconnect();
    el.innerHTML = "";
  }) as MountCleanup;
  cleanup.resizeObs = resizeObs;
  return cleanup;
};
