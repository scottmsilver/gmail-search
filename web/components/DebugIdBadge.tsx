"use client";

import { useState } from "react";

type Props = {
  data: { id: string; log_path: string };
};

export const DebugIdBadge = ({ data }: Props) => {
  const [copied, setCopied] = useState<"id" | "path" | null>(null);

  const copy = async (what: "id" | "path", value: string) => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(what);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      // ignore
    }
  };

  return (
    <div className="mt-2 flex items-center gap-2 text-[10px] text-neutral-400 font-mono">
      <span>log:</span>
      <button
        type="button"
        onClick={() => copy("id", data.id)}
        className="text-neutral-500 hover:text-neutral-800 underline-offset-2 hover:underline"
        title="Copy log id"
      >
        {data.id}
      </button>
      <span className="text-neutral-300">·</span>
      <button
        type="button"
        onClick={() => copy("path", data.log_path)}
        className="text-neutral-400 hover:text-neutral-700"
        title={`Copy path: ${data.log_path}`}
      >
        copy path
      </button>
      {copied && <span className="text-emerald-600">✓ copied {copied}</span>}
    </div>
  );
};
