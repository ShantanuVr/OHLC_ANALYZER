import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'OHLC Analyzer | SMC Backtesting Platform',
  description: 'No-Code Backtesting Platform for XAUUSD Trading with Smart Money Concepts',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  )
}

