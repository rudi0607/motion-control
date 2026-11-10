import { useRef, useState, type ReactNode } from "react";

export function DropZone(props: {
  accept: string;
  title: string;
  hint: string;
  icon: ReactNode;
  file: File | null;
  onFile: (f: File | null) => void;
  children?: ReactNode;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (f) props.onFile(f);
  };
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        pick(e.dataTransfer.files);
      }}
      className={`relative flex min-h-[220px] flex-col overflow-hidden rounded-xl border border-dashed transition ${
        over ? "border-accent bg-accent/10" : "border-ink-600 bg-ink-850 hover:border-ink-400"
      }`}
    >
      <input ref={input} type="file" accept={props.accept} className="hidden" onChange={(e) => pick(e.target.files)} />
      {props.file && props.children ? (
        <>
          <div className="flex-1">{props.children}</div>
          <div className="flex items-center justify-between gap-2 border-t border-ink-700 bg-ink-900 px-3 py-2 text-xs">
            <span className="truncate text-ink-300">
              {props.file.name} · {(props.file.size / 1e6).toFixed(1)} MB
            </span>
            <div className="flex shrink-0 gap-2">
              <button type="button" className="text-accent hover:underline" onClick={() => input.current?.click()}>
                Replace
              </button>
              <button type="button" className="text-ink-400 hover:text-red-400" onClick={() => props.onFile(null)}>
                Remove
              </button>
            </div>
          </div>
        </>
      ) : (
        <button type="button" onClick={() => input.current?.click()} className="flex flex-1 flex-col items-center justify-center gap-3 p-6 text-center">
          <div className="grid h-12 w-12 place-items-center rounded-2xl bg-ink-700 text-accent">{props.icon}</div>
          <div>
            <div className="font-medium text-slate-100">{props.title}</div>
            <div className="mt-1 text-xs text-ink-400">{props.hint}</div>
          </div>
          <span className="btn-ghost text-xs">Browse or drop file</span>
        </button>
      )}
    </div>
  );
}

export const ImageIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <circle cx="12" cy="8" r="4" />
    <path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" strokeLinecap="round" />
  </svg>
);
export const VideoIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
    <rect x="3" y="5" width="13" height="14" rx="2" />
    <path d="M16 10l5-3v10l-5-3" strokeLinejoin="round" />
  </svg>
);
