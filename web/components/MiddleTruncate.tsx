"use client";

import { useLayoutEffect, useRef, useState } from "react";

type Props = {
  text: string;
  className?: string;
  title?: string;
};

// Character-based middle-ellipsis. CSS can only do end-ellipsis
// (`text-overflow: ellipsis`), so we measure `scrollWidth` vs
// `clientWidth` after layout and trim the middle until the string
// fits. A binary search keeps this cheap (O(log n) reflows) and stays
// exact — a pure ratio estimate undershoots on prose with wide glyphs
// (emoji, CJK) and overshoots on narrow ones.
//
// Full text goes into the `title` attribute (unless overridden) so
// hover reveals the hidden middle.
export const MiddleTruncate = ({ text, className, title }: Props) => {
  const ref = useRef<HTMLSpanElement>(null);
  const [display, setDisplay] = useState(text);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;

    const fit = () => {
      // Put the full string in first. If it already fits, we're done.
      el.textContent = text;
      if (el.scrollWidth <= el.clientWidth + 1) {
        setDisplay(text);
        return;
      }
      // Binary search over the number of characters to KEEP.
      // We keep `head = ceil(k/2)` from the start and `tail = floor(k/2)`
      // from the end, plus a single "…" between them.
      let lo = 3; // minimum "A…Z"
      let hi = text.length;
      let best = lo;
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        const head = Math.ceil(mid / 2);
        const tail = Math.floor(mid / 2);
        const candidate = text.slice(0, head) + "…" + text.slice(text.length - tail);
        el.textContent = candidate;
        if (el.scrollWidth <= el.clientWidth + 1) {
          best = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      const head = Math.ceil(best / 2);
      const tail = Math.floor(best / 2);
      setDisplay(text.slice(0, head) + "…" + text.slice(text.length - tail));
    };

    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(el);
    return () => observer.disconnect();
  }, [text]);

  return (
    <span
      ref={ref}
      className={className}
      title={title ?? text}
      style={{
        display: "inline-block",
        maxWidth: "100%",
        whiteSpace: "nowrap",
        overflow: "hidden",
      }}
    >
      {display}
    </span>
  );
};
