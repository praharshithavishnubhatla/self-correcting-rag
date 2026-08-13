"use client";

import { useState } from "react";
import { uploadDocumentsBatch } from "@/lib/api";

type FileStatus = {
  name: string;
  state: "pending" | "uploading" | "done" | "error";
  message?: string;
};

const DOC_TYPES = [
  { value: "auto", label: "Auto-detect" },
  { value: "notes", label: "Notes" },
  { value: "cheatsheet", label: "Cheatsheet" },
  { value: "social_post", label: "Screenshot / social post" },
];

export default function UploadPanel({
  topic,
  onUploaded,
}: {
  topic: string;
  onUploaded: () => void;
}) {
  const [docType, setDocType] = useState("auto");
  const [statuses, setStatuses] = useState<FileStatus[]>([]);
  const [busy, setBusy] = useState(false);

  async function handleFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList);
    if (!files.length) return;

    setBusy(true);
    setStatuses(files.map((f) => ({ name: f.name, state: "uploading" })));

    try {
      const { results } = await uploadDocumentsBatch(files, topic || "general", docType);
      setStatuses(
        results.map((r) => ({
          name: r.filename,
          state: r.ok ? "done" : "error",
          message: r.ok ? `${r.chunks_added} chunks · ${r.doc_type}` : r.error,
        }))
      );
    } catch (e: any) {
      setStatuses(files.map((f) => ({ name: f.name, state: "error", message: e.message })));
    } finally {
      setBusy(false);
      onUploaded();
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <label className="text-xs text-zinc-500">Tag uploads as</label>
        <select
          value={docType}
          onChange={(e) => setDocType(e.target.value)}
          className="rounded-md border border-border bg-card px-2 py-1 text-xs text-zinc-300"
        >
          {DOC_TYPES.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>
      </div>

      <div
        className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-card p-6 text-center"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (e.dataTransfer.files?.length) handleFiles(e.dataTransfer.files);
        }}
      >
        <p className="text-sm text-zinc-400">
          Drop notes, cheatsheets, or saved screenshots here — multiple files at once are fine
        </p>
        <label className="cursor-pointer rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white">
          {busy ? "Uploading…" : "Browse files"}
          <input
            type="file"
            multiple
            className="hidden"
            accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg"
            disabled={busy}
            onChange={(e) => {
              if (e.target.files?.length) handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </label>
      </div>

      {statuses.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs">
          {statuses.map((s) => (
            <li key={s.name} className="flex items-center justify-between">
              <span className="text-zinc-400">{s.name}</span>
              <span
                className={
                  s.state === "done"
                    ? "text-emerald-400"
                    : s.state === "error"
                    ? "text-red-400"
                    : "text-zinc-500"
                }
              >
                {s.state === "uploading" ? "uploading…" : s.message || s.state}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
