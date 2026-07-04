import type { Metadata } from "next";
import { Rajdhani, JetBrains_Mono, Inter } from "next/font/google";
import "./globals.css";
import { LenisProvider } from "@/components/scroll/LenisProvider";

const rajdhani = Rajdhani({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-rajdhani",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Pitwall — AI Race Strategy",
  description:
    "7 agents. Live telemetry. Pit-wall grade answers. An AI strategy engine for F1 race weekends.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${rajdhani.variable} ${jetbrainsMono.variable} ${inter.variable}`}>
      <body className="bg-carbon text-text-primary font-body antialiased">
        <LenisProvider>{children}</LenisProvider>
      </body>
    </html>
  );
}
