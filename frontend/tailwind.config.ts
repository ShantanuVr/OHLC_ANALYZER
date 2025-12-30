import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'gold': {
          50: '#fffbeb',
          100: '#fef3c7',
          200: '#fde68a',
          300: '#fcd34d',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
          700: '#b45309',
          800: '#92400e',
          900: '#78350f',
        },
        'chart': {
          bg: '#0a0a0f',
          grid: '#1a1a2e',
          up: '#22c55e',
          down: '#ef4444',
          fvg: 'rgba(34, 197, 94, 0.15)',
          ob: 'rgba(168, 85, 247, 0.2)',
        }
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Fira Code', 'monospace'],
        'sans': ['Geist', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
export default config

