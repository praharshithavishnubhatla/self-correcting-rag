"use client";

export default function TopicSelector({
  topics,
  selected,
  onSelect,
  onCreateNew,
}: {
  topics: string[];
  selected: string;
  onSelect: (t: string) => void;
  onCreateNew: (t: string) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {topics.map((t) => (
        <button
          key={t}
          onClick={() => onSelect(t)}
          className={`rounded-full border px-3 py-1 text-xs ${
            selected === t
              ? "border-accent bg-accent/20 text-accent"
              : "border-border text-zinc-400"
          }`}
        >
          {t}
        </button>
      ))}
      <button
        onClick={() => {
          const name = prompt("New topic name (e.g. os, dbms, system-design)");
          if (name) onCreateNew(name.trim().toLowerCase());
        }}
        className="rounded-full border border-dashed border-border px-3 py-1 text-xs text-zinc-500"
      >
        + new topic
      </button>
    </div>
  );
}
