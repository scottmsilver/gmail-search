"use client";

import { useCallback, useEffect, useState } from "react";

import type { JobsRunningResponse, RunningJob } from "@/lib/backend";

const formatBytes = (n: number): string => {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${(n / 1024).toFixed(1)} KiB`;
};

const JobRow = ({ j }: { j: RunningJob }) => {
  const pct = j.total > 0 ? Math.min(100, Math.round((j.completed / j.total) * 100)) : null;
  return (
    <div
      className="flex items-center justify-between gap-4 px-3 py-2 rounded-md text-xs"
      style={{ background: "var(--bg-secondary)" }}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium" style={{ color: "var(--fg-primary)" }}>
            {j.job_id}
          </span>
          <span style={{ color: "var(--fg-tertiary)" }}>·</span>
          <span style={{ color: "var(--fg-secondary)" }}>{j.stage}</span>
          {j.status !== "running" && (
            <span className="uppercase tracking-wide" style={{ color: "var(--fg-tertiary)" }}>
              {j.status}
            </span>
          )}
        </div>
        <div className="truncate mt-0.5" style={{ color: "var(--fg-tertiary)" }}>
          {j.detail || "—"}
        </div>
        {pct !== null && (
          <div className="mt-1.5 h-1 rounded overflow-hidden" style={{ background: "var(--bg-primary)" }}>
            <div className="h-full" style={{ width: `${pct}%`, background: "var(--fg-secondary)" }} />
          </div>
        )}
      </div>
      {pct !== null && (
        <div className="text-[10px] tabular-nums shrink-0" style={{ color: "var(--fg-tertiary)" }}>
          {j.completed.toLocaleString()}/{j.total.toLocaleString()} ({pct}%)
        </div>
      )}
    </div>
  );
};

export const SettingsView = () => {
  const [data, setData] = useState<JobsRunningResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minFreeGb, setMinFreeGb] = useState(5);
  const [intervalSec, setIntervalSec] = useState(120);
  const [busy, setBusy] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/jobs/running", { cache: "no-store" });
      if (!res.ok) throw new Error(`status ${res.status}`);
      setData((await res.json()) as JobsRunningResponse);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 3_000);
    return () => clearInterval(id);
  }, [refresh]);

  const kickoff = useCallback(
    async (kind: "frontfill" | "backfill" | "frontfill-stop") => {
      setBusy(kind);
      setToast(null);
      try {
        let url: string;
        if (kind === "frontfill") {
          url = `/api/jobs/frontfill?interval=${encodeURIComponent(String(intervalSec))}`;
        } else if (kind === "frontfill-stop") {
          url = "/api/jobs/frontfill/stop";
        } else {
          url = `/api/jobs/backfill?min_free_gb=${encodeURIComponent(String(minFreeGb))}`;
        }
        const res = await fetch(url, { method: "POST" });
        const body = (await res.json()) as { ok: boolean; pid?: number; error?: string };
        if (!body.ok) {
          setToast(body.error ?? `failed: ${kind}`);
        } else if (kind === "frontfill-stop") {
          setToast(`watch stopped`);
          await refresh();
        } else {
          setToast(`${kind} started (pid ${body.pid})`);
          await refresh();
        }
      } catch (e) {
        setToast(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(null);
      }
    },
    [intervalSec, minFreeGb, refresh],
  );

  const running = data?.running ?? [];
  const recent = data?.recent ?? [];
  const disk = data?.disk;
  const frontfillRunning = data?.frontfill?.running ?? false;
  const frontfillPid = data?.frontfill?.pid ?? null;
  const backfillRunning = running.some((j) => j.job_id === "update");

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 space-y-8">
      <div>
        <h1 className="text-base font-medium" style={{ color: "var(--fg-primary)" }}>
          Settings
        </h1>
        <p className="text-xs mt-1" style={{ color: "var(--fg-tertiary)" }}>
          Background jobs and disk usage.
        </p>
      </div>

      {error && (
        <div className="text-xs px-3 py-2 rounded" style={{ color: "var(--fg-secondary)", background: "var(--bg-secondary)" }}>
          {error}
        </div>
      )}

      <section className="space-y-2">
        <h2 className="text-xs uppercase tracking-wide" style={{ color: "var(--fg-tertiary)" }}>
          Disk
        </h2>
        {disk ? (
          <div
            className="text-xs px-3 py-2 rounded"
            style={{ background: "var(--bg-secondary)", color: "var(--fg-secondary)" }}
          >
            <div className="flex justify-between">
              <span>Free</span>
              <span className="tabular-nums">{formatBytes(disk.free_bytes)}</span>
            </div>
            <div className="flex justify-between">
              <span>Used</span>
              <span className="tabular-nums">{formatBytes(disk.used_bytes)}</span>
            </div>
            <div className="flex justify-between">
              <span>Total</span>
              <span className="tabular-nums">{formatBytes(disk.total_bytes)}</span>
            </div>
          </div>
        ) : (
          <div className="text-xs" style={{ color: "var(--fg-tertiary)" }}>
            loading…
          </div>
        )}
      </section>

      <section className="space-y-2">
        <h2 className="text-xs uppercase tracking-wide" style={{ color: "var(--fg-tertiary)" }}>
          Running jobs
        </h2>
        {running.length === 0 ? (
          <div className="text-xs" style={{ color: "var(--fg-tertiary)" }}>
            none
          </div>
        ) : (
          <div className="space-y-1.5">
            {running.map((j) => (
              <JobRow key={j.job_id} j={j} />
            ))}
          </div>
        )}
      </section>

      <section className="space-y-3">
        <h2 className="text-xs uppercase tracking-wide" style={{ color: "var(--fg-tertiary)" }}>
          Actions
        </h2>
        <div className="space-y-3">
          <div className="px-3 py-3 rounded-md space-y-3" style={{ background: "var(--bg-secondary)" }}>
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-xs font-medium" style={{ color: "var(--fg-primary)" }}>
                  Frontfill
                </div>
                <div className="text-[11px] mt-0.5" style={{ color: "var(--fg-tertiary)" }}>
                  Continuously watch Gmail for new messages — sync, extract, embed, reindex every N seconds.
                  {frontfillRunning && frontfillPid !== null && (
                    <>
                      {" "}
                      <span style={{ color: "var(--fg-secondary)" }}>Running (pid {frontfillPid}).</span>
                    </>
                  )}
                </div>
              </div>
              {frontfillRunning ? (
                <button
                  type="button"
                  disabled={busy === "frontfill-stop"}
                  onClick={() => void kickoff("frontfill-stop")}
                  className="text-xs px-3 py-1.5 rounded-full font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: "var(--bg-primary)", color: "var(--fg-primary)", border: "1px solid var(--border-subtle)" }}
                >
                  {busy === "frontfill-stop" ? "stopping…" : "Stop"}
                </button>
              ) : (
                <button
                  type="button"
                  disabled={busy === "frontfill"}
                  onClick={() => void kickoff("frontfill")}
                  className="text-xs px-3 py-1.5 rounded-full font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: "var(--fg-primary)", color: "var(--bg-primary)" }}
                >
                  {busy === "frontfill" ? "starting…" : "Start"}
                </button>
              )}
            </div>
            {!frontfillRunning && (
              <label className="flex items-center gap-3 text-[11px]" style={{ color: "var(--fg-secondary)" }}>
                <span>Check every</span>
                <input
                  type="number"
                  min={10}
                  max={86400}
                  step={30}
                  value={intervalSec}
                  onChange={(e) => setIntervalSec(Math.max(10, Number(e.target.value) || 120))}
                  className="w-20 px-2 py-1 rounded text-right tabular-nums"
                  style={{
                    background: "var(--bg-primary)",
                    color: "var(--fg-primary)",
                    border: "1px solid var(--border-subtle)",
                  }}
                />
                <span>seconds</span>
              </label>
            )}
          </div>

          <div className="px-3 py-3 rounded-md space-y-3" style={{ background: "var(--bg-secondary)" }}>
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="text-xs font-medium" style={{ color: "var(--fg-primary)" }}>
                  Backfill
                </div>
                <div className="text-[11px] mt-0.5" style={{ color: "var(--fg-tertiary)" }}>
                  Pull older messages in batches. Stops automatically when free disk hits the floor below.
                </div>
              </div>
              <button
                type="button"
                disabled={busy === "backfill" || backfillRunning}
                onClick={() => void kickoff("backfill")}
                className="text-xs px-3 py-1.5 rounded-full font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ background: "var(--fg-primary)", color: "var(--bg-primary)" }}
              >
                {backfillRunning ? "running…" : busy === "backfill" ? "starting…" : "Run now"}
              </button>
            </div>
            <label className="flex items-center gap-3 text-[11px]" style={{ color: "var(--fg-secondary)" }}>
              <span>Stop when free disk drops below</span>
              <input
                type="number"
                min={1}
                max={10000}
                step={1}
                value={minFreeGb}
                onChange={(e) => setMinFreeGb(Math.max(1, Number(e.target.value) || 1))}
                className="w-16 px-2 py-1 rounded text-right tabular-nums"
                style={{
                  background: "var(--bg-primary)",
                  color: "var(--fg-primary)",
                  border: "1px solid var(--border-subtle)",
                }}
              />
              <span>GiB</span>
            </label>
          </div>
        </div>
        {toast && (
          <div className="text-[11px]" style={{ color: "var(--fg-tertiary)" }}>
            {toast}
          </div>
        )}
      </section>

      {recent.length > 0 && (
        <section className="space-y-2">
          <h2 className="text-xs uppercase tracking-wide" style={{ color: "var(--fg-tertiary)" }}>
            Recent
          </h2>
          <div className="space-y-1.5">
            {recent.map((j) => (
              <JobRow key={`${j.job_id}-${j.started_at}`} j={j} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
