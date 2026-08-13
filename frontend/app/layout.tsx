import "./globals.css";

export const metadata = {
  title: "Exam Prep RAG",
  description: "Grounded answers from your own notes, cheatsheets, and saved posts.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
