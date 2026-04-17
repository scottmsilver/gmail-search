"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

type Row = {
  variant: string;
  wins: number;
  losses: number;
  ties: number;
  both_bad: number;
  total: number;
  win_rate: number;
};

export default function StatsPage() {
  const [rows, setRows] = useState<Row[]>([]);
  const [battles, setBattles] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const res = await fetch("/api/battle/stats");
        if (!res.ok) throw new Error(`stats ${res.status}`);
        const data = (await res.json()) as { leaderboard: Row[]; battles: number };
        setRows(data.leaderboard ?? []);
        setBattles(data.battles ?? 0);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  return (
    <div className="max-w-3xl mx-auto p-6">
      <header className="mb-4 flex items-center gap-4">
        <Link href="/battle" className="text-sm text-neutral-500 hover:text-neutral-800">← battle</Link>
        <h1 className="text-xl font-semibold">Leaderboard</h1>
        <span className="text-sm text-neutral-500 ml-auto">{battles} battles recorded</span>
      </header>

      {err && <div className="text-red-600 text-sm">Error: {err}</div>}

      {rows.length === 0 && !err && (
        <div className="text-sm text-neutral-500">No battles yet. <Link href="/battle" className="text-blue-600 underline">Go start some</Link>.</div>
      )}

      {rows.length > 0 && (
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-left text-xs text-neutral-500 border-b">
              <th className="py-2">Variant</th>
              <th className="text-right">Win rate</th>
              <th className="text-right">W</th>
              <th className="text-right">L</th>
              <th className="text-right">T</th>
              <th className="text-right">BB</th>
              <th className="text-right">Total</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.variant} className="border-b">
                <td className="py-2 font-mono text-xs">{r.variant}</td>
                <td className="text-right font-medium">{(r.win_rate * 100).toFixed(0)}%</td>
                <td className="text-right text-emerald-700">{r.wins}</td>
                <td className="text-right text-red-700">{r.losses}</td>
                <td className="text-right text-neutral-500">{r.ties}</td>
                <td className="text-right text-neutral-400">{r.both_bad}</td>
                <td className="text-right text-neutral-500">{r.total}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
