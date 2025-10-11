import './globals.css'

export const metadata = {
  title: 'PriceOptima Dashboard',
  description: 'AI-Powered Dynamic Pricing Analytics Platform',
  keywords: 'pricing, analytics, AI, machine learning, optimization, revenue',
  authors: [{ name: 'PriceOptima Team' }],
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">{children}</body>
    </html>
  )
}
