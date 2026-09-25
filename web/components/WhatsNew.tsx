// "What's new" dialog (#73), invited/public web only. Opened from the account
// menu, and automatically once per new release: on load the newest release
// in the notes is compared with the one this browser last recorded, and the
// dialog pops only when an earlier one was recorded (never on a first visit).
// Native <dialog> + showModal() for the focus trap and Escape.

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  fetchWhatsNew, isFreshRelease, latestRelease, readSeenRelease, writeSeenRelease, type WhatsNewDoc,
} from "@/lib/whatsNew";
import { cn } from "@/lib/utils";

const browserStorage = (): Storage | undefined => {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
};

export function useWhatsNew(enabled: boolean) {
  const [doc, setDoc] = useState<WhatsNewDoc | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    void fetchWhatsNew().then((notes) => {
      const current = latestRelease(notes);
      if (cancelled || current === null) return;
      const storage = browserStorage();
      const fresh = isFreshRelease(current, readSeenRelease(storage));
      writeSeenRelease(storage, current);
      if (fresh) {
        setDoc(notes);
        setOpen(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [enabled]);

  const show = useCallback(() => {
    setDoc(null);
    setOpen(true);
    void fetchWhatsNew().then(setDoc);
  }, []);
  const close = useCallback(() => setOpen(false), []);
  return { doc, open, show, close };
}

export function WhatsNewDialog({ doc, open, onClose }: { doc: WhatsNewDoc | null; open: boolean; onClose: () => void }) {
  const ref = useRef<HTMLDialogElement>(null);

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="whats-new-title"
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClose={onClose}
      // A click on the backdrop lands on the dialog element itself.
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      className={cn(
        "w-[min(32rem,calc(100vw-2rem))] max-h-[80dvh] rounded-md border border-border bg-popover p-0",
        "text-popover-foreground shadow-lg backdrop:bg-black/40",
      )}
    >
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <h2 id="whats-new-title" className="text-sm font-semibold">What&apos;s new</h2>
        <button
          type="button"
          onClick={onClose}
          className="rounded px-2 py-1 text-xs transition hover:bg-accent hover:text-accent-foreground"
        >
          Close
        </button>
      </div>
      <div className="max-h-[calc(80dvh-3rem)] overflow-y-auto px-4 py-3 text-sm">
        <WhatsNewBody doc={doc} />
      </div>
    </dialog>
  );
}

function WhatsNewBody({ doc }: { doc: WhatsNewDoc | null }) {
  if (doc === null) return <p className="text-xs text-muted-foreground">Loading…</p>;
  if (!doc.releases.length) return <p className="text-xs text-muted-foreground">No release notes yet.</p>;
  return (
    <div className="space-y-4">
      {doc.releases.map((release) => (
        <section key={release.release}>
          <h3 className="mb-1 text-xs font-medium text-muted-foreground">{release.release}</h3>
          <ul className="list-disc space-y-1 pl-5">
            {release.issues.map((issue) => (
              <li key={issue.number}>{issue.title || `Issue #${issue.number}`}</li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
