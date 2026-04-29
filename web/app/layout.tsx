import type { Metadata } from "next";
import "./globals.css";

import { AuthGate } from "@/components/AuthGate";
import { PreviewDrawer } from "@/components/PreviewDrawer";
import { PreviewProvider } from "@/components/PreviewContext";
import { TopNav } from "@/components/TopNav";

export const metadata: Metadata = {
  title: "Gmail Search",
  description: "Chat with and search your Gmail archive",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="flex h-screen flex-col bg-background text-foreground antialiased">
        <AuthGate>
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
