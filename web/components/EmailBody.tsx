"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  textBody: string;
  htmlBody?: string;
};

// Render a sandboxed email body.
//
// Why iframe + sandbox: email HTML is attacker-controlled. We render
// inside a Gmail-style sandboxed iframe so:
//   * `<script>` tags never execute
//   * No `<form>` can submit to anywhere
//   * No top-level navigation, no window.opener tricks
//   * CSS is fully isolated from the app's styles
//
// We DO keep `allow-same-origin` set, which still blocks script
// execution but lets us measure the rendered height from the parent
// to auto-size the iframe to content. Without same-origin the
// document is opaque and we'd be stuck with a fixed viewport.
//
// Fallback behaviour:
//   * HTML missing → render text in a wrapped <pre>
//   * Text blank + HTML present → render HTML in the iframe
//   * Both present → iframe with a "view plain text" toggle
//
// No external lib — we could pull in DOMPurify + sanitize-html, but
// iframe isolation is strictly safer than any DOM-scrubbing approach
// for email (every major webmail does it this way).
export const EmailBody = ({ textBody, htmlBody }: Props) => {
  const hasHtml = !!(htmlBody && htmlBody.trim());
  const hasText = !!(textBody && textBody.trim());
  const [mode, setMode] = useState<"html" | "text">(hasHtml ? "html" : "text");

  if (mode === "text" || !hasHtml) {
    return (
      <div>
        {hasHtml && (
          <ModeToggle mode={mode} hasHtml={hasHtml} hasText={hasText} onChange={setMode} />
        )}
        {hasText ? (
          <pre className="mt-2 whitespace-pre-wrap font-sans text-sm leading-relaxed text-neutral-800">
            {textBody}
          </pre>
        ) : (
          <div className="mt-2 text-sm italic text-muted-foreground">(empty body)</div>
        )}
      </div>
    );
  }

  return (
    <div>
      <ModeToggle mode={mode} hasHtml={hasHtml} hasText={hasText} onChange={setMode} />
      <SandboxedHtml html={htmlBody!} />
    </div>
  );
};

const ModeToggle = ({
  mode,
  hasHtml,
  hasText,
  onChange,
}: {
  mode: "html" | "text";
  hasHtml: boolean;
  hasText: boolean;
  onChange: (m: "html" | "text") => void;
}) => {
  if (!hasHtml || !hasText) return null;
  return (
    <div className="mb-2 flex gap-1 text-[11px]">
      <button
        type="button"
        onClick={() => onChange("html")}
        className={`rounded px-2 py-0.5 ${
          mode === "html" ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-muted"
        }`}
      >
        Rendered
      </button>
      <button
        type="button"
        onClick={() => onChange("text")}
        className={`rounded px-2 py-0.5 ${
          mode === "text" ? "bg-secondary text-foreground" : "text-muted-foreground hover:bg-muted"
        }`}
      >
        Plain text
      </button>
    </div>
  );
};

// Read the current theme's surface + primary-text colors from the
// CSS vars we publish on :root. Re-reads on `data-theme` changes so
// switching to dark / sepia / slate repaints the email body.
//
// Why we touch the DOM at all: the iframe is isolated — CSS custom
// properties set on the parent document don't propagate into srcDoc.
// We have to compute them on the host and inline the values.
const useThemeColors = () => {
  const [colors, setColors] = useState({ bg: "#ffffff", fg: "#111111", accent: "#2563eb" });
  useEffect(() => {
    const read = () => {
      const cs = getComputedStyle(document.documentElement);
      setColors({
        bg: cs.getPropertyValue("--bg-surface").trim() || "#ffffff",
        fg: cs.getPropertyValue("--fg-primary").trim() || "#111111",
        accent: cs.getPropertyValue("--accent").trim() || "#2563eb",
      });
    };
    read();
    // Theme toggle flips the data-theme attribute on <html>; watch for
    // that and re-read so the email body re-skins immediately.
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme", "class"],
    });
    return () => obs.disconnect();
  }, []);
  return colors;
};

const SandboxedHtml = ({ html }: { html: string }) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [height, setHeight] = useState(300);
  const { bg, fg, accent } = useThemeColors();

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    let ro: ResizeObserver | null = null;

    const resize = () => {
      const doc = iframe.contentDocument;
      if (!doc || !doc.body) return;
      // `scrollHeight` is the unclipped content size. Add a few px so
      // we don't accidentally cut off baseline / overflow shadows.
      const h = Math.max(doc.body.scrollHeight + 8, 80);
      setHeight(h);
    };

    const onLoad = () => {
      resize();
      const doc = iframe.contentDocument;
      if (doc && doc.body) {
        ro = new ResizeObserver(resize);
        ro.observe(doc.body);
      }
    };

    iframe.addEventListener("load", onLoad);
    // In case the `load` event already fired before we attached.
    if (iframe.contentDocument?.readyState === "complete") onLoad();

    return () => {
      iframe.removeEventListener("load", onLoad);
      ro?.disconnect();
    };
  }, [html]);

  // Wrap the email's HTML in a minimal doc so its own styles apply and
  // there's a <base target="_blank"> so any link inside the email
  // opens in a new tab instead of trying to nav the iframe. Theme-aware
  // defaults — email HTML that doesn't set its own bg/color picks up
  // the active theme. Emails with inline styles (tables with baked-in
  // white cells, marketing templates) keep their own look, which is
  // the right call: rewriting them would break more than it fixes.
  const doc = `<!doctype html><html><head><base target="_blank">
<style>
  html, body { margin: 0; padding: 0; font: 14px/1.5 system-ui, -apple-system, sans-serif; color: ${fg}; background: ${bg}; word-break: break-word; }
  img { max-width: 100%; height: auto; }
  a { color: ${accent}; }
  table { max-width: 100%; }
</style>
</head><body>${html}</body></html>`;

  return (
    <iframe
      ref={iframeRef}
      srcDoc={doc}
      // `allow-same-origin` lets the PARENT measure content height.
      // No other capability is granted — scripts, forms, top-level
      // nav, popups, pointer lock, all off. This matches how Gmail
      // / Outlook render untrusted HTML.
      sandbox="allow-same-origin"
      title="Email body"
      className="w-full rounded border"
      // Match the iframe's body bg on the HOST side too — otherwise
      // there's a split-second white flash before srcDoc paints, which
      // is visible on dark / sepia / slate themes.
      style={{ backgroundColor: bg, height: `${height}px` }}
    />
  );
};
