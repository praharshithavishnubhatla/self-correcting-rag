/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        surface: "#0f0f10",
        card: "#18181b",
        border: "#2a2a2e",
        accent: "#6366f1",
      },
    },
  },
  plugins: [],
};
