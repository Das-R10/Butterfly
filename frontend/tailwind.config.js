/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        research: {
          bg: '#0A0E17',
          surface: '#111726',
          elevated: '#172033',
          border: '#1F293D',
          borderLight: '#2D3B55',
          textMuted: '#94A3B8',
          textMain: '#F1F5F9',
        },
        archetype: {
          analyst: '#38BDF8',    // Institutional Analyst (Cyan/Blue)
          debunker: '#F43F5E',   // Skeptic Debunker (Rose)
          seeker: '#A855F7',     // Pattern Seeker (Purple)
          doomer: '#F59E0B',     // Doomer (Amber/Orange)
          amplifier: '#10B981',  // Amplifier Influencer (Emerald)
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
