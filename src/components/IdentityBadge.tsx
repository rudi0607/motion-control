import type { QualityCheck } from "../api";

export type IdentityState =
  | { kind: "none" }
  | { kind: "analyzing" }
  | { kind: "no-face" }
  | { kind: "locked"; mode: "element" | "prompt" | "off" }
  | { kind: "verified"; check: QualityCheck };

export function IdentityBadge({ state }: { state: IdentityState }) {
  let tone = "border-ink-700 bg-ink-850 text-ink-300";
  let title = "Identity protection";
  let text = "Upload a character image to establish the identity reference.";
  if (state.kind === "analyzing") text = "Detecting face landmarks in the character image…";
  if (state.kind === "no-face") {
    tone = "border-red-500/40 bg-red-500/10 text-red-200";
    title = "No face detected";
    text = "Identity can't be locked or verified. Use a clear, unobstructed photo of the character's face.";
  }
  if (state.kind === "locked") {
    if (state.mode === "element") {
      tone = "border-emerald-500/40 bg-emerald-500/10 text-emerald-200";
      title = "Identity locked · element binding";
      text = "A face crop of your character is bound as a facial element. The reference video is used only as a motion source.";
    } else if (state.mode === "prompt") {
      tone = "border-amber-500/40 bg-amber-500/10 text-amber-200";
      title = "Identity locked · image + prompt";
      text = "This model/setting can't bind a facial element; identity comes from the character image with identity-preservation guidance.";
    } else {
      tone = "border-ink-600 bg-ink-850 text-ink-300";
      title = "Identity lock off";
      text = "The model relies on the character image only.";
    }
  }
  if (state.kind === "verified") {
    const s = state.check.status;
    tone =
      s === "pass"
        ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-200"
        : s === "warn"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-200"
          : s === "fail"
            ? "border-red-500/40 bg-red-500/10 text-red-200"
            : "border-ink-600 bg-ink-850 text-ink-300";
    title = `Identity check · ${s}${state.check.value ? ` · ${state.check.value}` : ""}`;
    text = state.check.detail;
  }
  return (
    <div className={`flex gap-3 rounded-xl border p-3 ${tone}`}>
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" className="mt-0.5 shrink-0">
        <path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" strokeLinejoin="round" />
        {state.kind === "locked" || (state.kind === "verified" && state.check.status === "pass") ? (
          <path d="M9 12l2 2 4-4" strokeLinecap="round" />
        ) : (
          <path d="M12 8v5M12 16h.01" strokeLinecap="round" />
        )}
      </svg>
      <div>
        <div className="text-sm font-medium">{title}</div>
        <div className="text-[11px] leading-snug opacity-80">{text}</div>
      </div>
    </div>
  );
}
