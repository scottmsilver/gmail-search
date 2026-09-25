import type { Metadata, Viewport } from "next";
import "./globals.css";

import { AuthGate } from "@/components/AuthGate";
import { PreviewDrawer } from "@/components/PreviewDrawer";
import { PreviewProvider } from "@/components/PreviewContext";
import { TopNav } from "@/components/TopNav";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Gmail Search",
  description: "Deep analysis of your Gmail archive",
};

// `interactiveWidget: "resizes-content"` makes the on-screen keyboard
// resize the layout viewport itself (not just the visual viewport), so
// the flex column below reflows around it instead of the composer
// ending up rendered off past the visible, keyboard-covered area.
// `viewportFit: "cover"` pairs with `env(safe-area-inset-*)` for devices
// with a notch/home-indicator. See body's `h-dvh` in this file.
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  interactiveWidget: "resizes-content",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      {/* `h-dvh` (100dvh), not `h-screen` (100vh): `vh` is fixed to the
          largest possible viewport on mobile Chrome and ignores the
          collapsing URL bar and the keyboard, which is how the composer
          at the bottom of this column ends up rendered below the fold. */}
      <body className="flex h-dvh flex-col bg-background text-foreground antialiased">
        <AuthGate publicMode={process.env.GMS_PUBLIC_ORIGIN !== undefined} fullWorkerMode={process.env.GMS_FULL_WORKER_ROUTES === "1"}>
          <PreviewProvider>
            <TopNav />
            <main className="flex-1 min-h-0">{children}</main>
            <PreviewDrawer />
          </PreviewProvider>
        </AuthGate>
      </body>
    </html>
  );
}
