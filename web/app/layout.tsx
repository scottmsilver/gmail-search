import type { Metadata } from "next";
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex h-screen flex-col bg-background text-foreground antialiased">
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
