"use client";

import { useState, type ReactNode } from "react";

// Pretty-print a JSON-ish value with subtle syntax highlighting and
// collapsible object / array nodes. No deps — react-json-view and
// react-json-tree are both ~30KB and bring their own theming model
// that fights our existing data-theme remaps. This module is ~150
// lines and does what we need: keys in blue, strings in green,
// numbers in amber, null/bool in rose, indented two spaces, with a
// caret on every nested object/array so deep payloads (full tool
// results, deep-mode evidence dumps) don't blow up the disclosure.
//
// Top-level objects/arrays render expanded; everything below auto-
// collapses by default. Click a caret to drill in. The wrapping
// `<pre>` keeps long primitive strings on one line — they overflow
// horizontally inside the container's existing scrollbar instead of
// breaking mid-token, which was the main legibility complaint.

type Props = {
  value: unknown;
  // Maximum string-value length before we clip. Beyond this we render
  // "…" + a (n chars) hint so the tree stays scannable without losing
  // the option to inspect (the raw value remains in the title attr).
  stringClipChars?: number;
};

const STRING_CLIP_DEFAULT = 240;

export const JsonView = ({ value, stringClipChars = STRING_CLIP_DEFAULT }: Props) => {
  return (
    <pre className="bg-neutral-50 border border-neutral-200 rounded p-2 overflow-x-auto text-[11px] leading-relaxed font-mono max-h-96 overflow-y-auto">
      <code>
        <Node value={value} depth={0} initiallyOpen stringClipChars={stringClipChars} />
      </code>
    </pre>
  );
};

const INDENT = "  ";

type NodeProps = {
  value: unknown;
  depth: number;
  initiallyOpen?: boolean;
  stringClipChars: number;
};

const Node = ({ value, depth, initiallyOpen = false, stringClipChars }: NodeProps): ReactNode => {
  if (value === null) return <span className="text-rose-600">null</span>;
  if (typeof value === "boolean") return <span className="text-rose-600">{String(value)}</span>;
  if (typeof value === "number") return <span className="text-amber-700">{value}</span>;
  if (typeof value === "string") return <StringValue value={value} clip={stringClipChars} />;
  if (Array.isArray(value))
    return <ArrayNode value={value} depth={depth} initiallyOpen={initiallyOpen} stringClipChars={stringClipChars} />;
  if (value && typeof value === "object")
    return (
      <ObjectNode
        value={value as Record<string, unknown>}
        depth={depth}
        initiallyOpen={initiallyOpen}
        stringClipChars={stringClipChars}
      />
    );
  return <span>{String(value)}</span>;
};

const StringValue = ({ value, clip }: { value: string; clip: number }) => {
  const escaped = escapeJsonString(value);
  if (escaped.length <= clip) {
    return <span className="text-emerald-700 break-words whitespace-pre-wrap">&quot;{escaped}&quot;</span>;
  }
  return (
    <span
      className="text-emerald-700 break-words whitespace-pre-wrap"
      title={value}
    >
      &quot;{escaped.slice(0, clip)}
      <span className="text-neutral-400">… ({escaped.length} chars)</span>
      &quot;
    </span>
  );
};

// Same escaping JSON.stringify does for the characters that matter
// inside a string literal — backslash, quote, newline, tab. Anything
// else passes through (we're rendering, not transmitting).
const escapeJsonString = (s: string): string =>
  s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n").replace(/\t/g, "\\t");

const ArrayNode = ({
  value,
  depth,
  initiallyOpen,
  stringClipChars,
}: {
  value: unknown[];
  depth: number;
  initiallyOpen: boolean;
  stringClipChars: number;
}) => {
  const [open, setOpen] = useState(initiallyOpen);
  if (value.length === 0) return <span>[]</span>;
  if (!open) {
    return (
      <span className="cursor-pointer text-neutral-400 hover:text-neutral-700" onClick={() => setOpen(true)}>
        [<span className="mx-1 underline decoration-dotted">{value.length} items</span>]
      </span>
    );
  }
  return (
    <span>
      <span
        className="cursor-pointer text-neutral-500 hover:text-neutral-800"
        onClick={() => setOpen(false)}
        title="Collapse"
      >
        [
      </span>
      {value.map((item, i) => (
        <span key={i}>
          {"\n" + INDENT.repeat(depth + 1)}
          <Node value={item} depth={depth + 1} stringClipChars={stringClipChars} />
          {i < value.length - 1 ? "," : ""}
        </span>
      ))}
      {"\n" + INDENT.repeat(depth)}
      <span
        className="cursor-pointer text-neutral-500 hover:text-neutral-800"
        onClick={() => setOpen(false)}
        title="Collapse"
      >
        ]
      </span>
    </span>
  );
};

const ObjectNode = ({
  value,
  depth,
  initiallyOpen,
  stringClipChars,
}: {
  value: Record<string, unknown>;
  depth: number;
  initiallyOpen: boolean;
  stringClipChars: number;
}) => {
  const [open, setOpen] = useState(initiallyOpen);
  const entries = Object.entries(value);
  if (entries.length === 0) return <span>{"{}"}</span>;
  if (!open) {
    return (
      <span className="cursor-pointer text-neutral-400 hover:text-neutral-700" onClick={() => setOpen(true)}>
        {"{"}
        <span className="mx-1 underline decoration-dotted">{entries.length} keys</span>
        {"}"}
      </span>
    );
  }
  return (
    <span>
      <span
        className="cursor-pointer text-neutral-500 hover:text-neutral-800"
        onClick={() => setOpen(false)}
        title="Collapse"
      >
        {"{"}
      </span>
      {entries.map(([k, v], i) => (
        <span key={k}>
          {"\n" + INDENT.repeat(depth + 1)}
          <span className="text-sky-700">&quot;{k}&quot;</span>
          <span className="text-neutral-500">: </span>
          <Node value={v} depth={depth + 1} stringClipChars={stringClipChars} />
          {i < entries.length - 1 ? "," : ""}
        </span>
      ))}
      {"\n" + INDENT.repeat(depth)}
      <span
        className="cursor-pointer text-neutral-500 hover:text-neutral-800"
        onClick={() => setOpen(false)}
        title="Collapse"
      >
        {"}"}
      </span>
    </span>
  );
};
