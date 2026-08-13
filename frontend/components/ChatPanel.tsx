"use client";

import { useEffect, useState } from "react";
import { askQuestionStream, AskResponse, Mode, StreamEvent } from "@/lib/api";

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "explain", label: "Explain", hint: "Ask about your notes…" },
  { value: "revise", label: "Revise", hint: "e.g. \"quick notes for tomorrow\" — topic material only, question is optional" },
  { value: "practice", label: "Practice", hint: "Generates practice Q&A from the topic — question is optional" },
];

type ChatTurn = { role: "user" | "assistant"; content: string };

function historyKey(topic: string | null) {
  return `rag-chat-history:${topic || "general"}`;
}

// Chat history is per-topic and persisted client-side only — the backend is
// stateless across requests and never stores conversation turns itself.
function loadHistory(topic: string | null): ChatTurn[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(historyKey(topic));
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(topic: string | null, history: ChatTurn[]) {
  try {
    window.localStorage.setItem(historyKey(topic), JSON.stringify(history));
  } catch {
    // localStorage full/unavailable — history just won't persist this time.
  }
}

// Only the last few turns are sent to the backend per request, to keep the
// prompt small — full history still lives in localStorage regardless.
const MAX_TURNS_SENT = 12;

export default function ChatPanel({ topic }: { topic: string | null }) {
  const [mode, setMode] = useState<Mode>("explain");
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [stageMessage, setStageMessage] = useState<string | null>(null);
  const [history, setHistory] = useState<ChatTurn[]>([]);
  const [lastResult, setLastResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Reload history whenever the topic changes.
  useEffect(() => {
    setHistory(loadHistory(topic));
    setLastResult(null);
    setError(null);
  }, [topic]);

  const activeMode = MODES.find((m) => m.value === mode)!;
  const needsTopic = mode !== "explain";

  async function handleAsk() {
    if (mode === "explain" && !question.trim()) return;
    if (needsTopic && !topic) {
      setError(`Pick a topic first — ${activeMode.label} covers a whole topic's material.`);
      return;
    }
    setLoading(true);
    setError(null);
    setStageMessage("Starting…");

    const askedQuestion = question;

    try {
      const historyToSend = mode === "explain" ? history.slice(-MAX_TURNS_SENT) : undefined;

      await askQuestionStream(askedQuestion, topic, mode, historyToSend, (event: StreamEvent) => {
        if (event.stage === "error") {
          setError(event.message);
          return;
        }
        if (event.stage === "done") {
          setLastResult(event);
          setStageMessage(null);

          if (mode === "explain") {
            const updated: ChatTurn[] = [
              ...history,
              { role: "user", content: askedQuestion },
              { role: "assistant", content: event.answer },
            ];
            setHistory(updated);
            saveHistory(topic, updated);
            setQuestion("");
          }
          return;
        }
        // Any in-progress stage — update the live status line.
        setStageMessage(event.message);
      });
    } catch (e: any) {
      setError(e.message);
      setStageMessage(null);
    } finally {
      setLoading(false);
      setStageMessage(null);
    }
  }

  function clearHistory() {
    setHistory([]);
    saveHistory(topic, []);
    setLastResult(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-1 rounded-lg border border-border bg-card p-1 text-xs">
        {MODES.map((m) => (
          <button
            key={m.value}
            onClick={() => {
              setMode(m.value);
              setLastResult(null);
              setError(null);
            }}
            className={
              "flex-1 rounded-md px-3 py-1.5 font-medium transition-colors " +
              (mode === m.value ? "bg-accent text-white" : "text-zinc-400 hover:text-zinc-200")
            }
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === "explain" && history.length > 0 && (
        <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
          <div className="flex items-center justify-between">
            <p className="text-xs uppercase tracking-wide text-zinc-500">
              Conversation · {topic || "general"}
            </p>
            <button onClick={clearHistory} className="text-xs text-zinc-500 hover:text-red-400">
              Clear history
            </button>
          </div>
          <div className="flex max-h-80 flex-col gap-3 overflow-y-auto">
            {history.map((turn, i) => (
              <div key={i} className={turn.role === "user" ? "text-right" : "text-left"}>
                <span
                  className={
                    "inline-block max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm " +
                    (turn.role === "user" ? "bg-accent text-white" : "bg-zinc-800 text-zinc-100")
                  }
                >
                  {turn.content}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex gap-2">
        <input
          className="flex-1 rounded-lg border border-border bg-card px-4 py-2 text-sm outline-none focus:border-accent"
          placeholder={activeMode.hint}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleAsk()}
        />
        <button
          onClick={handleAsk}
          disabled={loading}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
        >
          {loading ? "Working…" : activeMode.label}
        </button>
      </div>
      {loading && stageMessage && (
        <p className="-mt-2 flex items-center gap-2 text-xs text-accent">
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
          {stageMessage}
        </p>
      )}
      {needsTopic && !loading && (
        <p className="-mt-2 text-xs text-zinc-500">
          {activeMode.label} covers all material under "{topic || "no topic selected"}" — question above is optional context, not required.
        </p>
      )}

      {error && (
        <p className="rounded-lg border border-red-900 bg-red-950/40 px-4 py-2 text-sm text-red-300">
          {error}
        </p>
      )}

      {/* Revise/Practice results and the guardrail/evaluator debug line for
          the most recent explain turn — the running thread above already
          shows explain answers, so this only shows extra metadata for it. */}
      {lastResult && mode !== "explain" && (
        <div className="rounded-lg border border-border bg-card p-4">
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{lastResult.answer}</p>
          {lastResult.sources.length > 0 && (
            <div className="mt-4 border-t border-border pt-3">
              <p className="mb-1 text-xs uppercase tracking-wide text-zinc-500">Sources</p>
              <ul className="text-xs text-zinc-400">
                {lastResult.sources.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
      {lastResult && mode === "explain" && (
        <div className="flex gap-2 text-xs text-zinc-500">
          <span>guardrail: {lastResult.guardrail_passed ? "passed" : "failed"}</span>
          <span>·</span>
          <span>evaluator: {lastResult.evaluator_verdict}</span>
          {lastResult.sources.length > 0 && (
            <>
              <span>·</span>
              <span>sources: {lastResult.sources.join(", ")}</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}