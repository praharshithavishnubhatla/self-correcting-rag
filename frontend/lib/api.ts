const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Only required for the write endpoints (/ingest, /ingest/batch, /eval) — see
// api/main.py's require_api_key. Public read endpoints (/ask, /topics, etc.)
// don't need it. Note this is a NEXT_PUBLIC_ var, so it ends up in the client
// bundle: fine for a personal demo where you're the only uploader, but not a
// substitute for real per-user auth if this app ever has multiple users.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "";
const authHeaders: Record<string, string> = API_KEY ? { "X-API-Key": API_KEY } : {};

export type Mode = "explain" | "revise" | "practice";

export type ChatTurn = { role: "user" | "assistant"; content: string };

export type AskResponse = {
  answer: string;
  sources: string[];
  topic: string | null;
  mode: Mode;
  guardrail_passed: boolean | null;
  evaluator_verdict: string | null;
};

// mode="explain" answers `question` against the topic, optionally continuing
// a conversation via `history` (oldest turn first).
// mode="revise"/"practice" ignore `question`/`history` and cover the whole `topic`.
export async function askQuestion(
  question: string,
  topic: string | null,
  mode: Mode = "explain",
  history?: ChatTurn[]
): Promise<AskResponse> {
  const res = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, topic, mode, history }),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Ask failed: ${res.status}`);
  }
  return res.json();
}

export type StreamEvent =
  | { stage: "retrieve" | "rewrite" | "rerank" | "guardrail" | "generate" | "verify" | "gather" | "compress"; message: string }
  | ({ stage: "done" } & AskResponse)
  | { stage: "error"; message: string };

// Streaming version of askQuestion over Server-Sent Events. Calls onEvent
// for every stage as the pipeline progresses, and resolves once the final
// "done" (or "error") event has been delivered. Uses fetch + a manual
// ReadableStream reader rather than the browser's native EventSource,
// since EventSource only supports GET requests and can't send a JSON body.
export async function askQuestionStream(
  question: string,
  topic: string | null,
  mode: Mode,
  history: ChatTurn[] | undefined,
  onEvent: (event: StreamEvent) => void
): Promise<void> {
  const res = await fetch(`${API_BASE}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, topic, mode, history }),
  });

  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Ask failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || ""; // last chunk may be incomplete — keep for next read

    for (const raw of events) {
      const line = raw.trim();
      if (!line.startsWith("data:")) continue;
      try {
        const payload = JSON.parse(line.slice(5).trim());
        onEvent(payload);
      } catch {
        // malformed/partial event — skip rather than crash the stream
      }
    }
  }
}

export async function listTopics(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/topics`);
  if (!res.ok) throw new Error(`Topics failed: ${res.status}`);
  const data = await res.json();
  return data.topics;
}

export async function uploadDocument(
  file: File,
  topic: string,
  docType?: string
): Promise<any> {
  const form = new FormData();
  form.append("file", file);
  form.append("topic", topic);
  if (docType && docType !== "auto") form.append("doc_type", docType);

  const res = await fetch(`${API_BASE}/ingest`, { method: "POST", headers: authHeaders, body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export type BatchIngestResult = {
  filename: string;
  ok: boolean;
  source?: string;
  doc_type?: string;
  chunks_added?: number;
  error?: string;
};

export async function uploadDocumentsBatch(
  files: File[],
  topic: string,
  docType?: string
): Promise<{ results: BatchIngestResult[]; reindexed: boolean }> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  form.append("topic", topic);
  if (docType && docType !== "auto") form.append("doc_type", docType);

  const res = await fetch(`${API_BASE}/ingest/batch`, { method: "POST", headers: authHeaders, body: form });
  if (!res.ok) {
    const detail = await res.json().catch(() => null);
    throw new Error(detail?.detail || `Batch upload failed: ${res.status}`);
  }
  return res.json();
}

export async function runEval(topic: string | null): Promise<any> {
  const res = await fetch(`${API_BASE}/eval`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders },
    body: JSON.stringify({ topic, include_faithfulness: true }),
  });
  if (!res.ok) throw new Error(`Eval failed: ${res.status}`);
  return res.json();
}