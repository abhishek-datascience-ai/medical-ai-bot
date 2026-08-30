import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Medical-AI Bot",
  description: "Role-aware healthcare RAG assistant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
