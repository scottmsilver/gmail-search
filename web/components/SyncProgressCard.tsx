"use client";

// Initial-sync progress for the signed-in user. Two render modes:
//   <SyncProgressCard />            — full card (settings page)
//   <SyncProgressCard banner />     — slim dismissable strip (app pages),
//                                     renders nothing once sync is done.
// Polls /api/users/me/sync-status every 5s while syncing, backs off to
// 60s once ready (new mail keeps trickling; no need to hammer).

import { useEffect, useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

type SyncStatus = {
  state: "syncing" | "ready";
  sync_enabled: boolean;
  messages_synced: number;
  messages_total: number | null;
  messages_embedded: number;
  messages_summarized: number;
  rate_per_min: number | null;
  eta_minutes: number | null;
  cost_usd: number;
  jobs: { job: string; status: string; detail: string }[];
};

function fmtEta(minutes: number | null): string | null {
  if (minutes == null) return null;
  if (minutes < 2) return "under 2 minutes";
  if (minutes < 90) return `about ${Math.round(minutes)} minutes`;
  const h = minutes / 60;
  if (h < 36) return `about ${h.toFixed(h < 10 ? 1 : 0)} hours`;
  return `about ${Math.round(h / 24)} days`;
}

function fmtCount(n: number | null): string {
  return n == null ? "…" : n.toLocaleString();
}

export function SyncProgressCard({ banner = false }: { banner?: boolean }) {
  const [status, setStatus] = useState<SyncStatus | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let stopped = false;
    const tick = async () => {
      try {
        const res = await fetch("/api/users/me/sync-status", { cache: "no-store" });
        if (res.ok) setStatus(await res.json());
      } catch {
        // transient — keep the last status on screen
      }
      if (!stopped) {
        const delay = status?.state === "ready" ? 60_000 : 5_000;
        timer = setTimeout(tick, delay);
      }
    };
    tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.state]);

  if (!status) return null;
  const { messages_synced, messages_total, messages_embedded } = status;
  const pct = messages_total ? Math.min(100, (messages_synced / Math.max(messages_total, 1)) * 100) : null;
  const searchablePct =
    messages_total ? Math.min(100, (messages_embedded / Math.max(messages_total, 1)) * 100) : null;
  const eta = fmtEta(status.eta_minutes);
  const done = status.state === "ready";

  if (banner) {
    if (done || dismissed) return null;
    return (
      <div className="flex items-center gap-3 border-b bg-muted/50 px-4 py-2 text-xs text-muted-foreground">
        <div className="h-1.5 w-32 shrink-0 overflow-hidden rounded-full bg-secondary">
          <div
            className="h-full bg-primary transition-all"
            style={{ width: `${pct ?? 5}%` }}
          />
        </div>
        <span className="truncate">
          Importing your mailbox: {fmtCount(messages_synced)} of {fmtCount(messages_total)} emails
          {eta ? ` — ${eta} left` : ""}. Search works now on what&apos;s already in.
        </span>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          className="ml-auto shrink-0 text-muted-foreground hover:text-foreground"
          aria-label="Dismiss sync progress"
        >
          ×
        </button>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">
          {done ? "Mailbox synced" : "Importing your mailbox"}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>
              Downloaded {fmtCount(messages_synced)} of {fmtCount(messages_total)} emails
            </span>
            <span>{pct != null ? `${pct.toFixed(pct > 99 ? 1 : 0)}%` : "counting…"}</span>
          </div>
          <Progress value={pct ?? 0} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Searchable (indexed) {fmtCount(messages_embedded)}</span>
            <span>{searchablePct != null ? `${searchablePct.toFixed(0)}%` : ""}</span>
          </div>
          <Progress value={searchablePct ?? 0} />
        </div>
        <div className="text-xs text-muted-foreground">
          {done ? (
            <>All caught up — new mail is picked up automatically.</>
          ) : (
            <>
              {eta ? `${eta[0].toUpperCase()}${eta.slice(1)} remaining` : "Estimating time remaining…"}
              {status.rate_per_min ? ` · ${Math.round(status.rate_per_min).toLocaleString()} emails/min` : ""}
              {" · "}oldest mail imports last — search already works on everything downloaded so far.
            </>
          )}
          {status.cost_usd > 0 && <> · ${status.cost_usd.toFixed(2)} spent</>}
        </div>
      </CardContent>
    </Card>
  );
}
