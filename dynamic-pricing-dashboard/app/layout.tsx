export const metadata = {
  title: 'PriceOptima Dashboard',
  description: 'Dynamic Pricing Analytics Application',
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
