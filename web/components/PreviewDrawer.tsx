"use client";

import { Drawer } from "./Drawer";
import { FileViewer } from "./FileViewer";
import { usePreview } from "./PreviewContext";

// Right-side drawer that hosts the FileViewer. Wider than ThreadDrawer
// because PDFs + CSVs need the horizontal room; still responsive on
// narrow viewports via `w-full`.
export const PreviewDrawer = () => {
  const { target, closePreview } = usePreview();
  return (
    <Drawer
      open={target !== null}
      onClose={closePreview}
      title={target?.filename ?? "Preview"}
      subtitle={target ? `${target.kind ?? "file"} · ${target.mimeType}` : undefined}
      widthClass="w-full sm:w-[56vw] md:w-[50vw] lg:w-[45vw] max-w-4xl"
    >
      {target && (
        <FileViewer
          url={target.url}
          filename={target.filename}
          mimeType={target.mimeType}
        />
      )}
    </Drawer>
  );
};
