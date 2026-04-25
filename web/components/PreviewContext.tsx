"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";

// What the caller hands to openPreview. `url` is a same-origin URL the
// drawer can fetch bytes from; `mimeType` picks the renderer; `filename`
// is for the header + download anchor. `kind` is a hint so the drawer
// can badge "Artifact" vs "Attachment" vs "File" without us needing to
// parse the URL.
export type PreviewTarget = {
  url: string;
  filename: string;
  mimeType: string;
  kind?: "artifact" | "attachment" | "file";
};

type Ctx = {
  target: PreviewTarget | null;
  openPreview: (t: PreviewTarget) => void;
  closePreview: () => void;
};

const PreviewCtx = createContext<Ctx | null>(null);

// Kept in component state (not URL) on purpose — preview is transient;
// linking to a preview can be a follow-up if we want bookmarkable URLs.
export const PreviewProvider = ({ children }: { children: ReactNode }) => {
  const [target, setTarget] = useState<PreviewTarget | null>(null);
  const openPreview = useCallback((t: PreviewTarget) => setTarget(t), []);
  const closePreview = useCallback(() => setTarget(null), []);
  const value = useMemo(
    () => ({ target, openPreview, closePreview }),
    [target, openPreview, closePreview],
  );
  return <PreviewCtx.Provider value={value}>{children}</PreviewCtx.Provider>;
};

export const usePreview = (): Ctx => {
  const ctx = useContext(PreviewCtx);
  if (!ctx) throw new Error("usePreview must be used inside PreviewProvider");
  return ctx;
};
