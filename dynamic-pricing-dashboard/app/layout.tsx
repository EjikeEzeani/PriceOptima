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
              window.addEventListener('unhandledrejection', function(event) {
                if (event.reason && typeof event.reason === 'string' && 
                    (event.reason.includes('MetaMask') || event.reason.includes('ethereum'))) {
                  console.warn('Suppressed Web3/MetaMask error:', event.reason);
                  event.preventDefault();
                }
              });
              
              // Prevent any Web3 provider injection errors
              if (typeof window !== 'undefined') {
                window.addEventListener('error', function(event) {
                  if (event.message && (event.message.includes('MetaMask') || event.message.includes('ethereum'))) {
                    console.warn('Suppressed Web3 error:', event.message);
                    event.preventDefault();
                  }
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
