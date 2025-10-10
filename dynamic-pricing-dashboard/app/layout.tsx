export const metadata = {
  title: 'PriceOptima',
  description: 'Dynamic Pricing Dashboard'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
