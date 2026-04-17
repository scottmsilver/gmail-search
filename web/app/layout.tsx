import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gmail Search — Ask",
  description: "Chat with your Gmail archive",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased text-neutral-900">{children}</body>
    </html>
  );
}
