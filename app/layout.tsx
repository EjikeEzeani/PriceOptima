import type React from "react"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
})

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              // Hint to some wallets to avoid injecting dapp detection scripts
              try { (window as any).__disableDappDetectionInjection = true } catch {}

              window.addEventListener('unhandledrejection', function(event) {
                try {
                  const r = (event as any).reason;
                  const msg = typeof r === 'string' ? r : (r && (r.message || r.toString())) || '';
                  if (msg && (msg.includes('MetaMask') || msg.includes('ethereum') || msg.includes('Failed to connect to MetaMask'))) {
                    console.warn('Suppressed Web3/MetaMask error:', msg);
                    event.preventDefault();
                    return false;
                  }
                } catch {}
              });
              
              // Prevent any Web3 provider injection errors
              if (typeof window !== 'undefined') {
                window.addEventListener('error', function(event) {
                  try {
                    const msg = (event as any).message || ((event as any).error && (event as any).error.message) || '';
                    if (msg && (msg.includes('MetaMask') || msg.includes('ethereum') || msg.includes('Failed to connect to MetaMask'))) {
                      console.warn('Suppressed Web3 error:', msg);
                      event.preventDefault();
                      return false;
                    }
                  } catch {}
                });
              }
            `,
          }}
        />
      </head>
      <body className="min-h-screen bg-background font-sans">{children}</body>
    </html>
  )
}

export const metadata = {
      generator: 'v0.app'
    };
