import type { Metadata } from "next";
import "./globals.css";

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
      <body className="antialiased text-neutral-900 flex flex-col h-screen">
        <TopNav />
        <main className="flex-1 min-h-0">{children}</main>
      </body>
    </html>
  );
}
