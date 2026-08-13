"use client";

import { useEffect, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import UploadPanel from "@/components/UploadPanel";
import TopicSelector from "@/components/TopicSelector";
import { listTopics } from "@/lib/api";

export default function Home() {
  const [topics, setTopics] = useState<string[]>([]);
  const [selectedTopic, setSelectedTopic] = useState<string>("general");

  async function refreshTopics() {
    try {
      const t = await listTopics();
      setTopics(t.length ? t : ["general"]);
    } catch {
      setTopics(["general"]);
    }
  }

  useEffect(() => {
    refreshTopics();
  }, []);

  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 px-4 py-10">
      <header>
        <h1 className="text-lg font-medium">Exam prep RAG</h1>
        <p className="text-sm text-zinc-500">
          Grounded answers from your own notes, cheatsheets, and saved posts.
        </p>
      </header>

      <section className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-wide text-zinc-500">Topic</p>
        <TopicSelector
          topics={topics}
          selected={selectedTopic}
          onSelect={setSelectedTopic}
          onCreateNew={(t) => {
            setTopics((prev) => Array.from(new Set([...prev, t])));
            setSelectedTopic(t);
          }}
        />
      </section>

      <section className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-wide text-zinc-500">Add material</p>
        <UploadPanel topic={selectedTopic} onUploaded={refreshTopics} />
      </section>

      <section className="flex flex-col gap-2">
        <p className="text-xs uppercase tracking-wide text-zinc-500">Ask</p>
        <ChatPanel topic={selectedTopic} />
      </section>
    </main>
  );
}
