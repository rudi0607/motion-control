import type { AppConfig, GenerationSettings, ModelInfo } from "../api";

function Toggle(props: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; badge?: string }) {
  return (
    <label className={`flex items-start justify-between gap-3 py-1.5 ${props.disabled ? "opacity-50" : ""}`}>
      <span>
        <span className="text-sm text-slate-200">{props.label}</span>
        {props.badge && <span className="chip ml-2 bg-ink-700 text-ink-300">{props.badge}</span>}
        {props.hint && <span className="block text-[11px] leading-snug text-ink-400">{props.hint}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={props.checked}
        aria-label={props.label}
        disabled={props.disabled}
        onClick={() => props.onChange(!props.checked)}
        className={`relative mt-0.5 h-5 w-9 shrink-0 rounded-full transition ${props.checked ? "bg-accent" : "bg-ink-600"}`}
      >
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition ${props.checked ? "left-4.5" : "left-0.5"}`} />
      </button>
    </label>
  );
}

function Seg<T extends string | number>(props: { value: T; options: { v: T; label: string; note?: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="grid auto-cols-fr grid-flow-col gap-1 rounded-xl bg-ink-850 p-1">
      {props.options.map((o) => (
        <button
          key={String(o.v)}
          type="button"
          title={o.note}
          onClick={() => props.onChange(o.v)}
          className={`rounded-lg px-2 py-1.5 text-xs font-medium transition ${
            props.value === o.v ? "bg-ink-600 text-white shadow" : "text-ink-300 hover:text-slate-100"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SettingsPanel(props: {
  config: AppConfig;
  settings: GenerationSettings;
  onChange: (s: GenerationSettings) => void;
  referenceDuration: number | null;
}) {
  const { settings: s, config } = props;
  const set = <K extends keyof GenerationSettings>(k: K, v: GenerationSettings[K]) => props.onChange({ ...s, [k]: v });
  const model: ModelInfo | undefined = config.models.find((m) => m.id === s.modelId);
  const caps = model?.capabilities;
  const maxDur = model ? (s.cameraMotionTransfer ? model.maxDuration.cameraOn : model.maxDuration.cameraOff) : 0;
  const tooLong = props.referenceDuration !== null && props.referenceDuration > maxDur + 0.05;
  const elementActive = !!caps?.identityElement.supported && s.identityLock && !s.cameraMotionTransfer;
  const noPrompt = !caps?.promptGuidance.supported;

  return (
    <div className="space-y-5">
      <section>
        <div className="panel-title mb-2">Motion model</div>
        <div className="space-y-1.5">
          {config.models.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => set("modelId", m.id)}
              className={`w-full rounded-xl border p-3 text-left transition ${
                m.id === s.modelId ? "border-accent bg-accent/10" : "border-ink-700 bg-ink-850 hover:border-ink-600"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm font-medium text-slate-100">{m.label}</span>
                {m.available ? (
                  <span className="chip bg-emerald-500/15 text-emerald-300">ready</span>
                ) : (
                  <span className="chip bg-amber-500/15 text-amber-300">needs {m.provider === "fal" ? "FAL_KEY" : "REPLICATE_API_TOKEN"}</span>
                )}
              </div>
              {m.id === s.modelId && (
                <div className="mt-1 text-[11px] leading-snug text-ink-300">
                  {m.description}
                  <div className="mt-1 text-ink-400">
                    Native {m.nativeResolution} · {m.pricing} · {m.endpoint}
                  </div>
                </div>
              )}
            </button>
          ))}
        </div>
      </section>

      <section className="space-y-3">
        <div className="panel-title">Output</div>
        <div>
          <div className="mb-1 text-xs text-ink-300">Resolution</div>
          <Seg
            value={s.resolution}
            onChange={(v) => set("resolution", v)}
            options={[
              { v: "720p", label: "720p" },
              { v: "1080p", label: "1080p" },
              { v: "2k", label: "2K", note: "Lanczos upscale of native output" },
              { v: "4k", label: "4K", note: "Lanczos upscale of native output" },
            ]}
          />
          {(s.resolution === "2k" || s.resolution === "4k") && (
            <p className="mt-1 text-[11px] text-amber-300/90">Models generate at ≤1080p; {s.resolution.toUpperCase()} is a resampled upscale, not new detail.</p>
          )}
        </div>
        <div>
          <div className="mb-1 text-xs text-ink-300">Frame rate</div>
          <Seg value={s.fps} onChange={(v) => set("fps", v)} options={[24, 30, 50, 60].map((v) => ({ v: v as GenerationSettings["fps"], label: `${v} fps` }))} />
          <p className="mt-1 text-[11px] text-ink-400">{caps?.nativeFps.how}</p>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-200">Video duration</span>
          <span className="text-xs text-ink-300">
            Match reference{props.referenceDuration !== null ? ` · ${props.referenceDuration.toFixed(1)} s` : ""} (max {maxDur} s)
          </span>
        </div>
        {tooLong && (
          <Toggle
            label="Trim to model limit"
            hint={`Reference exceeds ${maxDur} s. Enable to use only the first ${maxDur} s.`}
            checked={s.trimToModelLimit}
            onChange={(v) => set("trimToModelLimit", v)}
          />
        )}
      </section>

      <section className="space-y-1">
        <div className="panel-title mb-1">Identity</div>
        <Toggle
          label="Face Preservation & Identity Consistency"
          badge={s.identityLock ? "Maximum" : "Off"}
          hint={elementActive ? "Facial element binding + identity prompt." : s.identityLock ? "Identity prompt + character image (element binding unavailable for this model/setting)." : "Model relies on the image alone."}
          checked={s.identityLock}
          onChange={(v) => set("identityLock", v)}
        />
        <Toggle label="Facial Detail Enhancement" hint="Blinking, expression, teeth & lips guidance." checked={s.faceDetail} onChange={(v) => set("faceDetail", v)} disabled={noPrompt} badge="prompt" />
        <Toggle label="Hand Detail Enhancement" hint="Five-finger, natural-joint guidance." checked={s.handDetail} onChange={(v) => set("handDetail", v)} disabled={noPrompt} badge="prompt" />
        <Toggle label="Background Preservation" hint="Keep the character image's background." checked={s.backgroundPreservation} onChange={(v) => set("backgroundPreservation", v)} disabled={noPrompt} badge="prompt" />
        {noPrompt && <p className="text-[11px] text-amber-300/90">{caps?.promptGuidance.how}</p>}
      </section>

      <section className="space-y-2">
        <div className="panel-title">Motion</div>
        <div>
          <div className="mb-1 text-xs text-ink-300">Motion Accuracy</div>
          <Seg value={s.motionAccuracy} onChange={(v) => set("motionAccuracy", v)} options={[{ v: "maximum", label: "Maximum" }, { v: "balanced", label: "Balanced (faster)" }]} />
          <p className="mt-1 text-[11px] text-ink-400">{caps?.qualityControls.how}</p>
        </div>
        <div className="opacity-60">
          <div className="flex justify-between text-xs text-ink-300">
            <span>Motion Strength</span>
            <span>{s.motionStrength}%</span>
          </div>
          <input type="range" min={0} max={100} value={s.motionStrength} disabled className="w-full" onChange={(e) => set("motionStrength", Number(e.target.value))} />
          <p className="text-[11px] text-ink-400">{caps?.motionStrength.how}</p>
        </div>
        <Toggle
          label="Camera Motion Transfer"
          hint={caps?.cameraMotion.how}
          checked={s.cameraMotionTransfer}
          disabled={!caps?.cameraMotion.supported}
          onChange={(v) => set("cameraMotionTransfer", v)}
        />
        <Toggle
          label="Natural Motion Smoothing"
          hint="Motion-compensated frame interpolation when raising frame rate."
          checked={s.motionSmoothing}
          onChange={(v) => set("motionSmoothing", v)}
        />
        <Toggle label="Auto-refine flagged frames" hint="Re-render with temporal deflicker if flicker is detected." checked={s.autoRefine} onChange={(v) => set("autoRefine", v)} />
        <Toggle label="Keep reference audio" hint={caps?.audio.how} checked={s.keepAudio} onChange={(v) => set("keepAudio", v)} />
      </section>

      <section className="space-y-2">
        <div className="panel-title">Scene guidance (optional)</div>
        <textarea
          rows={2}
          className="w-full resize-none"
          placeholder="e.g. Indoor studio, soft daylight. Describe context, not motion."
          value={s.scenePrompt}
          maxLength={600}
          disabled={noPrompt}
          onChange={(e) => set("scenePrompt", e.target.value)}
        />
        {caps?.seed.supported && (
          <label className="flex items-center justify-between text-xs text-ink-300">
            Seed (blank = random)
            <input
              type="number"
              className="w-32"
              value={s.seed ?? ""}
              onChange={(e) => set("seed", e.target.value === "" ? null : Number(e.target.value))}
            />
          </label>
        )}
      </section>
    </div>
  );
}
