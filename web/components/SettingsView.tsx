"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { JobsRunningResponse, RunningJob } from "@/lib/backend";

const formatBytes = (n: number): string => {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GiB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MiB`;
  return `${(n / 1024).toFixed(1)} KiB`;
};

// Shared card for frontfill + backfill — both have the same shape
// (start/stop + one numeric input). Rendering in one place keeps their
// affordances identical.
const JobControlCard = ({
  title,
  description,
  running,
  pid,
  starting,
  stopping,
  onStart,
  onStop,
  inputId,
  inputLabel,
  inputUnit,
  inputMin,
  inputMax,
  inputStep,
  inputValue,
  onInputChange,
  alwaysShowInput = false,
}: {
  title: string;
  description: string;
  running: boolean;
  pid: number | null;
  starting: boolean;
  stopping: boolean;
  onStart: () => void;
  onStop: () => void;
  inputId: string;
  inputLabel: string;
  inputUnit: string;
  inputMin: number;
  inputMax: number;
  inputStep: number;
  inputValue: number;
  onInputChange: (v: number) => void;
  alwaysShowInput?: boolean;
}) => (
  <Card>
    <CardHeader className="flex-row items-start justify-between gap-4 space-y-0 pb-4">
      <div className="space-y-1">
        <CardTitle className="text-sm">{title}</CardTitle>
        <CardDescription>
          {description}
          {running && pid !== null && (
            <span className="ml-1 font-medium text-foreground">Running (pid {pid}).</span>
          )}
        </CardDescription>
      </div>
      {running ? (
        <Button variant="outline" size="sm" disabled={stopping} onClick={onStop}>
          {stopping ? "Stopping…" : "Stop"}
        </Button>
      ) : (
        <Button size="sm" disabled={starting} onClick={onStart}>
          {starting ? "Starting…" : "Start"}
        </Button>
      )}
    </CardHeader>
    {(alwaysShowInput || !running) && (
      <CardContent className="pt-0">
        <div className="flex items-center gap-3">
          <Label htmlFor={inputId} className="whitespace-nowrap text-xs text-muted-foreground">
            {inputLabel}
          </Label>
          <Input
            id={inputId}
            type="number"
            min={inputMin}
            max={inputMax}
            step={inputStep}
            value={inputValue}
            onChange={(e) => onInputChange(Number(e.target.value))}
            className="w-24 text-right tabular-nums"
          />
          <span className="text-xs text-muted-foreground">{inputUnit}</span>
        </div>
      </CardContent>
    )}
  </Card>
);

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

  type Kick = "frontfill" | "frontfill-stop" | "backfill" | "backfill-stop";

  const kickoff = useCallback(
    async (kind: Kick) => {
      const urls: Record<Kick, string> = {
        frontfill: `/api/jobs/frontfill?interval=${encodeURIComponent(String(intervalSec))}`,
        "frontfill-stop": "/api/jobs/frontfill/stop",
        backfill: `/api/jobs/backfill?min_free_gb=${encodeURIComponent(String(minFreeGb))}`,
        "backfill-stop": "/api/jobs/backfill/stop",
      };
      setBusy(kind);
      setToast(null);
      try {
        const res = await fetch(urls[kind], { method: "POST" });
        const body = (await res.json()) as { ok: boolean; pid?: number; error?: string };
        if (!body.ok) {
          setToast(body.error ?? `failed: ${kind}`);
        } else {
          setToast(kind.endsWith("-stop") ? `${kind.split("-")[0]} stopped` : `${kind} started (pid ${body.pid})`);
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

  const recent = data?.recent ?? [];
  const disk = data?.disk;
  const frontfillRunning = data?.frontfill?.running ?? false;
  const frontfillPid = data?.frontfill?.pid ?? null;
  const backfillRunning = data?.backfill?.running ?? false;
  const backfillPid = data?.backfill?.pid ?? null;

  return (
    <div className="mx-auto max-w-2xl space-y-6 px-6 py-8">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground">Background jobs and disk usage.</p>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Disk</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1.5 text-sm">
          {disk ? (
            <>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Free</span>
                <span className="tabular-nums">{formatBytes(disk.free_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Used</span>
                <span className="tabular-nums">{formatBytes(disk.used_bytes)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total</span>
                <span className="tabular-nums">{formatBytes(disk.total_bytes)}</span>
              </div>
            </>
          ) : (
            <div className="text-muted-foreground">loading…</div>
          )}
        </CardContent>
      </Card>

      <JobControlCard
        title="Frontfill"
        description="Continuously watch Gmail for new messages — sync, extract, embed, reindex every N seconds."
        running={frontfillRunning}
        pid={frontfillPid}
        starting={busy === "frontfill"}
        stopping={busy === "frontfill-stop"}
        onStart={() => void kickoff("frontfill")}
        onStop={() => void kickoff("frontfill-stop")}
        inputId="frontfill-interval"
        inputLabel="Check every"
        inputUnit="seconds"
        inputMin={10}
        inputMax={86400}
        inputStep={30}
        inputValue={intervalSec}
        onInputChange={(v) => setIntervalSec(Math.max(10, v || 120))}
      />

      <JobControlCard
        title="Backfill"
        description="Pull older messages in batches. Stops automatically when free disk hits the floor below."
        running={backfillRunning}
        pid={backfillPid}
        starting={busy === "backfill"}
        stopping={busy === "backfill-stop"}
        onStart={() => void kickoff("backfill")}
        onStop={() => void kickoff("backfill-stop")}
        inputId="backfill-min-free"
        inputLabel="Stop when free disk <"
        inputUnit="GiB"
        inputMin={1}
        inputMax={10000}
        inputStep={1}
        inputValue={minFreeGb}
        onInputChange={(v) => setMinFreeGb(Math.max(1, v || 1))}
        alwaysShowInput
      />

      {toast && <div className="text-xs text-muted-foreground">{toast}</div>}

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Recent jobs</CardTitle>
          <CardDescription>Most recent runs from the job_progress log.</CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          {recent.length === 0 ? (
            <div className="p-6 pt-0 text-sm text-muted-foreground">No job history yet.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="px-3 py-2 font-medium">Job</th>
                    <th className="px-3 py-2 font-medium">Stage</th>
                    <th className="px-3 py-2 font-medium">Status</th>
                    <th className="px-3 py-2 text-right font-medium">Progress</th>
                    <th className="px-3 py-2 font-medium">Detail</th>
                    <th className="px-3 py-2 text-right font-medium">Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {recent.map((j) => (
                    <JobTableRow key={`${j.job_id}-${j.started_at}`} j={j} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

const STATUS_BADGE: Record<string, string> = {
  running: "bg-primary/15 text-primary",
  done: "bg-muted text-foreground",
  stopped: "bg-muted text-muted-foreground",
  error: "bg-destructive/15 text-destructive",
};

const formatDuration = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  if (seconds < 86400) {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return m === 0 ? `${h}h` : `${h}h ${m}m`;
  }
  const d = Math.floor(seconds / 86400);
  const h = Math.round((seconds % 86400) / 3600);
  return h === 0 ? `${d}d` : `${d}d ${h}h`;
};

const formatRelative = (iso: string): string => {
  const dt = new Date(iso);
  if (isNaN(dt.getTime())) return "?";
  const diffSec = Math.floor((Date.now() - dt.getTime()) / 1000);
  if (diffSec < 60) return `${diffSec}s ago`;
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  const sameYear = dt.getFullYear() === new Date().getFullYear();
  return dt.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    ...(sameYear ? {} : { year: "numeric" }),
  });
};

const JobTableRow = ({ j }: { j: RunningJob }) => {
  const pct = j.total > 0 ? Math.min(100, Math.round((j.completed / j.total) * 100)) : null;
  const statusClass = STATUS_BADGE[j.status] ?? "bg-muted text-muted-foreground";
  const eta = j.eta_seconds !== undefined ? formatDuration(j.eta_seconds) : null;
  const rate = j.rate_per_sec !== undefined ? j.rate_per_sec : null;
  return (
    <tr className="border-b last:border-0">
      <td className="whitespace-nowrap px-3 py-2 font-medium text-foreground">{j.job_id}</td>
      <td className="whitespace-nowrap px-3 py-2 text-muted-foreground">{j.stage}</td>
      <td className="whitespace-nowrap px-3 py-2">
        <span className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-medium ${statusClass}`}>
          {j.status}
        </span>
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right tabular-nums">
        <div className="text-muted-foreground">
          {pct === null ? "—" : `${j.completed.toLocaleString()}/${j.total.toLocaleString()} (${pct}%)`}
        </div>
        {eta && (
          <div className="text-[10px] text-foreground">
            ~{eta} left
            {rate !== null && <span className="text-muted-foreground"> · {Math.round(rate * 60)}/min</span>}
          </div>
        )}
      </td>
      <td className="max-w-[200px] truncate px-3 py-2 text-muted-foreground" title={j.detail}>
        {j.detail || "—"}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right text-muted-foreground">
        {formatRelative(j.updated_at)}
      </td>
    </tr>
  );
};
