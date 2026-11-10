import type { Job, QualityCheck } from "../api";

const dot: Record<string, string> = {
  pending: "bg-ink-600",
  running: "bg-accent animate-pulse",
  done: "bg-emerald-400",
  failed: "bg-red-500",
  skipped: "bg-ink-500",
};

function fmt(ms: number) {
  const s = Math.round(ms / 1000);
  return s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`;
}

export function PipelinePanel({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const done = job.stages.filter((s) => s.status === "done" || s.status === "skipped").length;
  const pct = job.status === "succeeded" ? 100 : Math.round((done / job.stages.length) * 100);
  const running = job.stages.find((s) => s.status === "running");
  return (
    <div className="space-y-4">
      <div>
        <div className="mb-1.5 flex items-center justify-between text-xs">
          <span className="text-ink-300">
            {job.status === "running" ? running?.label ?? "Working…" : job.status === "succeeded" ? "Completed" : job.status === "canceled" ? "Canceled" : "Failed"}
          </span>
          <span className="tabular-nums text-ink-300">{pct}%</span>
        </div>
        <div className="h-2 overflow-hidden rounded-full bg-ink-700">
          <div
            className={`h-full rounded-full transition-all duration-500 ${job.status === "failed" ? "bg-red-500" : "bg-gradient-to-r from-accent to-accent-2"}`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-1.5 text-[11px] text-ink-400">Progress reflects completed pipeline stages and live provider status; generation time is set by the provider's GPU queue.</p>
      </div>
      <ol className="space-y-2">
        {job.stages.map((s, i) => (
          <li key={s.id} className="flex gap-3">
            <div className="flex flex-col items-center">
              <span className={`mt-1 h-2.5 w-2.5 rounded-full ${dot[s.status]}`} />
              {i < job.stages.length - 1 && <span className="mt-1 w-px flex-1 bg-ink-700" />}
            </div>
            <div className="min-w-0 flex-1 pb-1">
              <div className="flex items-center justify-between gap-2 text-sm">
                <span className={s.status === "pending" ? "text-ink-400" : "text-slate-100"}>
                  <span className="mr-1.5 text-ink-400">{i + 1}.</span>
                  {s.label}
                </span>
                {s.startedAt && (
                  <span className="text-[11px] tabular-nums text-ink-400">{fmt((s.endedAt ?? Date.now()) - s.startedAt)}</span>
                )}
              </div>
              {s.detail && <p className={`mt-0.5 break-words text-[11px] leading-snug ${s.status === "failed" ? "text-red-300" : "text-ink-300"}`}>{s.detail}</p>}
            </div>
          </li>
        ))}
      </ol>
      {job.provider.logs.length > 0 && (
        <details className="rounded-lg border border-ink-700 bg-ink-850 p-2 text-[11px]">
          <summary className="cursor-pointer text-ink-300">Provider logs ({job.provider.logs.length})</summary>
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-ink-400">{job.provider.logs.slice(-60).join("\n")}</pre>
        </details>
      )}
      {job.provider.sentInput && (
        <details className="rounded-lg border border-ink-700 bg-ink-850 p-2 text-[11px]">
          <summary className="cursor-pointer text-ink-300">Parameters sent to {job.provider.endpoint}</summary>
          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap text-ink-400">
            {JSON.stringify(job.provider.sentInput, (k, v) => (typeof v === "string" && /^https?:/.test(v) ? "<uploaded file>" : v), 2)}
          </pre>
        </details>
      )}
      {job.status === "running" && (
        <button type="button" className="btn-ghost w-full" onClick={onCancel}>
          Cancel generation
        </button>
      )}
    </div>
  );
}

const statusStyle: Record<QualityCheck["status"], string> = {
  pass: "bg-emerald-500/15 text-emerald-300",
  warn: "bg-amber-500/15 text-amber-300",
  fail: "bg-red-500/15 text-red-300",
  skipped: "bg-ink-700 text-ink-300",
};

export function QualityReport(props: { checks: QualityCheck[]; title: string; onSeek?: (t: number) => void }) {
  return (
    <div>
      <div className="panel-title mb-2">{props.title}</div>
      <ul className="space-y-1.5">
        {props.checks.map((c) => (
          <li key={c.id} className="rounded-lg border border-ink-700 bg-ink-850 p-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm text-slate-100">{c.label}</span>
              <span className={`chip ${statusStyle[c.status]}`}>{c.status}</span>
            </div>
            {c.value && <div className="mt-0.5 text-xs tabular-nums text-slate-300">{c.value}</div>}
            <p className="mt-0.5 text-[11px] leading-snug text-ink-400">{c.detail}</p>
            {!!c.ranges?.length && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {c.ranges.slice(0, 12).map((r, i) => (
                  <button
                    key={i}
                    type="button"
                    onClick={() => props.onSeek?.(r.start)}
                    className="chip bg-ink-700 text-[10px] text-slate-300 hover:bg-ink-600"
                  >
                    {r.start.toFixed(2)}–{Number.isFinite(r.end) ? r.end.toFixed(2) : "end"}s
                  </button>
                ))}
                {c.ranges.length > 12 && <span className="text-[10px] text-ink-400">+{c.ranges.length - 12} more</span>}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
