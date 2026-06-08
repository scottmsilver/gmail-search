// Admin-only desired-state sync console.
//
// The mental model: each account has ONE intent — the "Sync" toggle
// (sync_enabled). The supervisor is the engine that makes reality match
// intent: it keeps the three daemons (frontfill / backfill / summarize)
// alive for every synced account and stops them for paused ones. So the
// page reads top-down as "is the engine running?" then, per account,
// "what's the intent and does reality match?".
//
// Health dots are READ-ONLY status, not buttons:
//   🟢 running   🟡 should be running but isn't   ⚪ paused (sync off)
// Manual per-daemon start/stop lives under "details" for debugging a
// stuck daemon — it's not the primary control.

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useAuth } from "@/components/AuthContext";
import { Progress } from "@/components/ui/progress";

type DaemonStatus = {
  running: boolean;
  pid: number | null;
  age_seconds: number | null;
  stage: string;
  detail: string;
};

type GmailStatus = { connected: boolean; problem: "scope" | null };

type AdminUser = {
  id: string;
  email: string;
  name: string | null;
  sync_enabled: boolean;
  invited_at: string | null;
  last_login_at: string | null;
  msg_count: number;
  emb_count: number;
  frontfill: DaemonStatus;
  backfill: DaemonStatus;
  summarize: DaemonStatus;
  reindex: DaemonStatus;
  gmail?: GmailStatus;
};

type AdminPayload = { users: AdminUser[]; supervisor: DaemonStatus };

type EmbedRow = {
  id: string;
  email: string;
  messages: number;
  embedded: number;
  pending: number;
  pct: number;
};
type CrawlStats = {
  fast: number;
  slow: number;
  dead: number;
  done: number;
  pending: number;
  max_attempts: number;
};
type ProgressPayload = { embedding: EmbedRow[]; crawl: CrawlStats };

const POLL_MS = 4_000;
// The progress query scans the URL-stub table, so poll it slower than the
// lightweight daemon-health poll.
const PROGRESS_POLL_MS = 15_000;
const DAEMONS = [
  { key: "frontfill", label: "Frontfill", blurb: "new mail" },
  { key: "backfill", label: "Backfill", blurb: "older mail" },
  { key: "summarize", label: "Summarize", blurb: "AI summaries" },
  { key: "reindex", label: "Reindex", blurb: "search index" },
] as const;

const fmtAge = (sec: number | null): string => {
  if (sec === null) return "—";
  if (sec < 60) return `${sec.toFixed(0)}s ago`;
  if (sec < 3600) return `${(sec / 60).toFixed(0)}m ago`;
  return `${(sec / 3600).toFixed(1)}h ago`;
};

const fmtCount = (n: number): string =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1000 ? `${(n / 1000).toFixed(0)}k` : `${n}`;

const useAdminUsers = () => {
  const [data, setData] = useState<AdminPayload | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await fetch("/api/admin/users", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData((await res.json()) as AdminPayload);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  return { data, err, refresh };
};

type Eta = { embedMin: number | null; crawlMin: number | null; embedRate: number; crawlRate: number };

const useAdminProgress = () => {
  const [data, setData] = useState<ProgressPayload | null>(null);
  // Baseline (first sample this session) → cumulative-average rate. Stabler
  // than a per-poll delta and converges as the page stays open.
  const base = useRef<{ embedded: number; done: number; t: number } | null>(null);
  const [eta, setEta] = useState<Eta>({ embedMin: null, crawlMin: null, embedRate: 0, crawlRate: 0 });

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const res = await fetch("/api/admin/progress", { cache: "no-store" });
        if (!res.ok || !alive) return;
        const p = (await res.json()) as ProgressPayload;
        setData(p);
        const embedded = p.embedding.reduce((s, r) => s + r.embedded, 0);
        const pending = p.embedding.reduce((s, r) => s + r.pending, 0);
        const now = Date.now();
        if (!base.current) {
          base.current = { embedded, done: p.crawl.done, t: now };
          return;
        }
        const dtMin = (now - base.current.t) / 60_000;
        if (dtMin < 0.4) return; // need a little elapsed time for a stable rate
        const embedRate = (embedded - base.current.embedded) / dtMin;
        const crawlRate = (p.crawl.done - base.current.done) / dtMin;
        setEta({
          embedRate,
          crawlRate,
          embedMin: embedRate > 0 ? pending / embedRate : null,
          crawlMin: crawlRate > 0 ? p.crawl.pending / crawlRate : null,
        });
      } catch {
        /* keep showing the last good snapshot */
      }
    };
    void refresh();
    const id = setInterval(refresh, PROGRESS_POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);
  return { data, eta };
};

// "~2h 15m" / "~40 min" / "—". Used for both queues.
const fmtEta = (min: number | null): string => {
  if (min === null || !isFinite(min) || min <= 0) return "—";
  if (min < 1) return "<1 min";
  if (min < 60) return `~${Math.round(min)} min`;
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return m ? `~${h}h ${m}m` : `~${h}h`;
};

const fmtRate = (perMin: number): string =>
  perMin >= 1 ? `${Math.round(perMin).toLocaleString()}/min` : perMin > 0 ? `${Math.round(perMin * 60)}/hr` : "—";

// ── Work-remaining: embedding coverage + URL-crawl lanes ────────────────────
const EmbeddingProgress = ({
  rows,
  etaMin,
  ratePerMin,
}: {
  rows: EmbedRow[];
  etaMin: number | null;
  ratePerMin: number;
}) => {
  const totalPending = rows.reduce((s, r) => s + r.pending, 0);
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm font-medium">Embedding coverage</div>
        {totalPending > 0 ? (
          <div className="text-[11px] tabular-nums text-neutral-500">
            {ratePerMin > 0 ? `${fmtRate(ratePerMin)} · ` : ""}
            {fmtEta(etaMin)} left
          </div>
        ) : (
          <div className="text-[11px] font-medium text-emerald-700">all caught up</div>
        )}
      </div>
      <div className="text-[11px] text-neutral-500">
        Messages that are embedded (semantically searchable). The backfill fills the rest in bounded
        chunks; new mail is embedded by the frontfill.
      </div>
      <div className="mt-3 space-y-3">
        {rows.map((r) => (
          <div key={r.id}>
            <div className="flex items-baseline justify-between gap-2 text-[11px] tabular-nums">
              <span className="truncate text-neutral-600">{r.email}</span>
              <span className={r.pending > 0 ? "shrink-0 text-amber-700" : "shrink-0 text-emerald-700"}>
                {r.embedded.toLocaleString()} / {r.messages.toLocaleString()} ({r.pct}%)
                {r.pending > 0 ? ` · ${r.pending.toLocaleString()} left` : " · complete"}
              </span>
            </div>
            <Progress value={r.pct} className="mt-1" />
          </div>
        ))}
      </div>
    </div>
  );
};

const Lane = ({ color, label, n, sub }: { color: string; label: string; n: number; sub: string }) => (
  <div className="flex items-center gap-1.5 rounded-md border border-neutral-200 px-2 py-1.5">
    <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />
    <div className="leading-tight">
      <div className="font-medium text-neutral-700">{n.toLocaleString()}</div>
      <div className="text-[10px] text-neutral-500">
        {label} · {sub}
      </div>
    </div>
  </div>
);

const CrawlerCard = ({
  crawl,
  etaMin,
  ratePerMin,
}: {
  crawl: CrawlStats;
  etaMin: number | null;
  ratePerMin: number;
}) => {
  const total = crawl.fast + crawl.slow + crawl.dead + crawl.done || 1;
  const seg = (n: number) => `${((100 * n) / total).toFixed(1)}%`;
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4">
      <div className="flex items-baseline justify-between gap-2">
        <div className="text-sm font-medium">URL crawler</div>
        {crawl.pending > 0 ? (
          <div className="text-[11px] tabular-nums text-neutral-500">
            {ratePerMin > 0 ? `${fmtRate(ratePerMin)} · ${fmtEta(etaMin)} left` : "waiting for embed pass"}
          </div>
        ) : (
          <div className="text-[11px] font-medium text-emerald-700">queue clear</div>
        )}
      </div>
      <div className="text-[11px] text-neutral-500">
        Fetches linked pages from mail so their text is searchable. Fast lane = never tried (crawled
        first); slow lane = failed, retrying with backoff; abandoned after {crawl.max_attempts} tries
        so dead/anti-bot links can&apos;t head-of-line block live ones.
      </div>
      <div className="mt-3 flex h-2 w-full overflow-hidden rounded-full bg-neutral-100">
        <div className="bg-emerald-500" style={{ width: seg(crawl.done) }} />
        <div className="bg-blue-500" style={{ width: seg(crawl.fast) }} />
        <div className="bg-amber-500" style={{ width: seg(crawl.slow) }} />
        <div className="bg-neutral-400" style={{ width: seg(crawl.dead) }} />
      </div>
      <div className="mt-2 grid grid-cols-2 gap-1.5 text-[11px] tabular-nums sm:grid-cols-4">
        <Lane color="bg-blue-500" label="Fast lane" n={crawl.fast} sub="never tried" />
        <Lane color="bg-amber-500" label="Slow lane" n={crawl.slow} sub="retrying" />
        <Lane color="bg-neutral-400" label="Abandoned" n={crawl.dead} sub="dead links" />
        <Lane color="bg-emerald-500" label="Done" n={crawl.done} sub="crawled" />
      </div>
      <div className="mt-2 text-[11px] text-neutral-500 tabular-nums">
        {crawl.pending.toLocaleString()} pending to crawl
      </div>
    </div>
  );
};

// ── Supervisor banner — is the engine running? ──────────────────────────────
const SupervisorBanner = ({
  supervisor,
  busy,
  onStart,
  onStop,
}: {
  supervisor: DaemonStatus;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
}) => {
  const up = supervisor.running;
  return (
    <div
      className={`flex items-center justify-between gap-3 rounded-lg border px-4 py-3 ${
        up ? "border-emerald-200 bg-emerald-50" : "border-amber-300 bg-amber-50"
      }`}
    >
      <div className="flex items-center gap-2.5 min-w-0">
        <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${up ? "bg-emerald-500" : "bg-amber-500"}`} />
        <div className="min-w-0">
          <div className="text-sm font-medium">
            Supervisor {up ? "running" : "stopped"}
          </div>
          <div className="text-xs text-neutral-600 truncate">
            {up
              ? supervisor.detail || "keeping each synced account's daemons alive"
              : "Desired state is NOT being enforced — daemons won't auto-start or self-heal."}
          </div>
        </div>
      </div>
      <button
        type="button"
        onClick={up ? onStop : onStart}
        disabled={busy}
        className={`shrink-0 rounded-md px-3 py-1.5 text-xs font-medium disabled:opacity-50 ${
          up
            ? "border border-neutral-300 text-neutral-700 hover:bg-white"
            : "bg-amber-600 text-white hover:bg-amber-700"
        }`}
      >
        {busy ? "…" : up ? "Stop" : "Start supervisor"}
      </button>
    </div>
  );
};

// ── One read-only health dot ────────────────────────────────────────────────
const HealthDot = ({
  label,
  blurb,
  desired,
  status,
}: {
  label: string;
  blurb: string;
  desired: boolean;
  status: DaemonStatus;
}) => {
  // green = running; amber = should run but isn't; grey = paused.
  const state = !desired ? "off" : status.running ? "on" : "down";
  const color =
    state === "on" ? "bg-emerald-500" : state === "down" ? "bg-amber-500" : "bg-neutral-300";
  const word = state === "on" ? "running" : state === "down" ? "down" : "off";
  const tip =
    state === "on"
      ? `${status.stage || "running"} · ${fmtAge(status.age_seconds)}${status.detail ? ` · ${status.detail}` : ""}`
      : state === "down"
        ? "should be running — supervisor will (re)start it"
        : "paused (sync off)";
  return (
    <div
      className="flex items-center gap-2 rounded-md border border-neutral-200 px-2.5 py-2"
      title={tip}
    >
      <span className={`h-2 w-2 shrink-0 rounded-full ${color}`} />
      <div className="min-w-0 leading-tight">
        <div className="text-xs font-medium">{label}</div>
        <div className="text-[10px] text-neutral-500">
          {word} · {blurb}
        </div>
      </div>
    </div>
  );
};

// ── Gmail connection chip + (re)connect ─────────────────────────────────────
const GmailChip = ({ gmail, isSelf }: { gmail: GmailStatus | undefined; isSelf: boolean }) => {
  if (!gmail) return null;
  if (gmail.connected) {
    return (
      <span className="rounded-full bg-emerald-500/15 px-2 py-0.5 text-[11px] text-emerald-700">
        Gmail connected
      </span>
    );
  }
  const reason = gmail.problem === "scope" ? "missing Gmail scope" : "Gmail not connected";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="rounded-full bg-amber-500/15 px-2 py-0.5 text-[11px] text-amber-700">{reason}</span>
      {isSelf ? (
        <a
          href="/api/auth/connect-gmail?return_url=%2Fadmin"
          className="text-[11px] font-medium text-blue-600 hover:underline"
        >
          Reconnect
        </a>
      ) : (
        <span className="text-[11px] text-neutral-400">owner must reconnect</span>
      )}
    </span>
  );
};

// ── Simple on/off switch ────────────────────────────────────────────────────
const Switch = ({
  on,
  disabled,
  onChange,
}: {
  on: boolean;
  disabled: boolean;
  onChange: (v: boolean) => void;
}) => (
  <button
    type="button"
    role="switch"
    aria-checked={on}
    disabled={disabled}
    onClick={() => onChange(!on)}
    className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:opacity-50 ${
      on ? "bg-emerald-500" : "bg-neutral-300"
    }`}
  >
    <span
      className={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
        on ? "translate-x-4" : "translate-x-0.5"
      }`}
    />
  </button>
);

const AccountCard = ({
  u,
  selfEmail,
  busyKey,
  onSetSync,
  onDaemon,
}: {
  u: AdminUser;
  selfEmail: string | null;
  busyKey: string | null;
  onSetSync: (uid: string, enabled: boolean) => void;
  onDaemon: (uid: string, kind: string, action: "start" | "stop") => void;
}) => {
  const [showDetails, setShowDetails] = useState(false);
  const syncBusy = busyKey === `${u.id}:sync`;
  return (
    <div className="rounded-lg border border-neutral-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-medium">
            {u.name ?? u.email.split("@")[0]}
            <span className="ml-1.5 font-normal text-neutral-500">{u.email}</span>
          </div>
          <div className="mt-0.5 text-[11px] text-neutral-500 tabular-nums">
            {fmtCount(u.msg_count)} msgs · {fmtCount(u.emb_count)} embeddings
            {u.last_login_at ? ` · last sign-in ${u.last_login_at.slice(0, 10)}` : ""}
          </div>
          <div className="mt-1.5">
            <GmailChip gmail={u.gmail} isSelf={!!selfEmail && selfEmail === u.email} />
          </div>
        </div>
        <label className="flex shrink-0 items-center gap-2 text-xs font-medium text-neutral-700">
          <span className={u.sync_enabled ? "text-emerald-700" : "text-neutral-400"}>
            {u.sync_enabled ? "Sync on" : "Sync off"}
          </span>
          <Switch on={u.sync_enabled} disabled={syncBusy} onChange={(v) => onSetSync(u.id, v)} />
        </label>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-4">
        {DAEMONS.map((d) => (
          <HealthDot
            key={d.key}
            label={d.label}
            blurb={d.blurb}
            desired={u.sync_enabled}
            status={u[d.key]}
          />
        ))}
      </div>

      <div className="mt-2">
        <button
          type="button"
          onClick={() => setShowDetails((s) => !s)}
          className="text-[11px] text-neutral-500 hover:text-neutral-800"
        >
          {showDetails ? "▾ hide controls" : "▸ details"}
        </button>
        {showDetails && (
          <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-4">
            {DAEMONS.map((d) => {
              const status = u[d.key];
              const startBusy = busyKey === `${u.id}:${d.key}:start`;
              const stopBusy = busyKey === `${u.id}:${d.key}:stop`;
              return (
                <div key={d.key} className="flex items-center gap-1">
                  <span className="w-16 text-[11px] text-neutral-500">{d.label}</span>
                  <button
                    type="button"
                    onClick={() => onDaemon(u.id, d.key, "start")}
                    disabled={status.running || startBusy}
                    className="flex-1 rounded border border-neutral-200 px-2 py-1 text-[11px] text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
                  >
                    start
                  </button>
                  <button
                    type="button"
                    onClick={() => onDaemon(u.id, d.key, "stop")}
                    disabled={!status.running || stopBusy}
                    className="flex-1 rounded border border-neutral-200 px-2 py-1 text-[11px] text-neutral-600 hover:bg-neutral-50 disabled:opacity-40"
                  >
                    stop
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

export function AdminConsole() {
  const { user } = useAuth();
  const { data, err, refresh } = useAdminUsers();
  const { data: progress, eta } = useAdminProgress();
  const [busy, setBusy] = useState<string | null>(null);

  const act = useCallback(
    async (key: string, fn: () => Promise<unknown>) => {
      setBusy(key);
      try {
        await fn();
        await refresh();
      } finally {
        setBusy(null);
      }
    },
    [refresh],
  );

  const setSync = (uid: string, enabled: boolean) =>
    act(`${uid}:sync`, () =>
      fetch(`/api/admin/users/${uid}/sync_enabled`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled }),
      }),
    );

  const daemon = (uid: string, kind: string, action: "start" | "stop") =>
    act(`${uid}:${kind}:${action}`, () =>
      fetch(`/api/admin/users/${uid}/${kind}/${action}`, { method: "POST" }),
    );

  const supervisor = (action: "start" | "stop") =>
    act(`supervisor:${action}`, () => fetch(`/api/admin/supervisor/${action}`, { method: "POST" }));

  if (!user?.is_admin) {
    return (
      <div className="mx-auto max-w-2xl px-6 py-12 text-sm">
        <h1 className="mb-2 text-lg font-semibold">Admin only</h1>
        <p className="text-neutral-600">
          This page is restricted to admins (set via the
          <code className="mx-1 rounded bg-neutral-100 px-1">GMS_ADMIN_EMAILS</code>
          env var on the server).
        </p>
      </div>
    );
  }

  const supBusy = busy === "supervisor:start" || busy === "supervisor:stop";

  return (
    <div className="mx-auto max-w-3xl px-6 py-8">
      <header className="mb-5">
        <h1 className="text-xl font-semibold">Sync</h1>
        <p className="text-xs text-neutral-500">
          Flip <span className="font-medium">Sync</span> per account; the supervisor makes reality
          match. Dots are live status, not buttons.
        </p>
      </header>

      {err ? (
        <div className="mb-4 rounded border border-red-200 bg-red-50 p-3 text-sm text-red-800">
          Couldn&apos;t load: {err}
        </div>
      ) : null}

      {data ? (
        <SupervisorBanner
          supervisor={data.supervisor}
          busy={supBusy}
          onStart={() => supervisor("start")}
          onStop={() => supervisor("stop")}
        />
      ) : (
        <div className="text-sm text-neutral-400">Loading…</div>
      )}

      <div className="mt-4 space-y-3">
        {data?.users.length === 0 ? (
          <div className="rounded border border-neutral-200 p-6 text-sm text-neutral-500">
            No users enrolled yet.
          </div>
        ) : null}
        {data?.users.map((u) => (
          <AccountCard
            key={u.id}
            u={u}
            selfEmail={user?.email ?? null}
            busyKey={busy}
            onSetSync={setSync}
            onDaemon={daemon}
          />
        ))}
      </div>

      {progress ? (
        <div className="mt-6 space-y-3">
          <h2 className="text-sm font-semibold text-neutral-700">Work remaining</h2>
          <EmbeddingProgress rows={progress.embedding} etaMin={eta.embedMin} ratePerMin={eta.embedRate} />
          <CrawlerCard crawl={progress.crawl} etaMin={eta.crawlMin} ratePerMin={eta.crawlRate} />
        </div>
      ) : null}

      {data ? (
        <div className="mt-5 flex items-center gap-4 text-[11px] text-neutral-500">
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-emerald-500" /> running
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-amber-500" /> should be running
          </span>
          <span className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-neutral-300" /> paused
          </span>
        </div>
      ) : null}
    </div>
  );
}
