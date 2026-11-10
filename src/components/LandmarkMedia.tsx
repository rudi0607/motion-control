import { useEffect, useRef, type RefObject } from "react";
import { frameAt, type AnalysisResult, type FrameLandmarks } from "../vision/landmarks";
import { containBox, drawLandmarks, type Layers } from "../vision/draw";

/** Video or image with a live landmark overlay drawn on a canvas above it. */
export function LandmarkMedia(props: {
  kind: "video" | "image";
  src: string;
  analysis?: AnalysisResult | null;
  frame?: FrameLandmarks | null;
  layers: Layers;
  videoRef?: RefObject<HTMLVideoElement | null>;
  controls?: boolean;
  muted?: boolean;
  className?: string;
  label?: string;
}) {
  const localVideo = useRef<HTMLVideoElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const vref = props.videoRef ?? localVideo;
  const propsRef = useRef(props);
  propsRef.current = props;

  // While extracting, show the exact frame whose landmarks are drawn.
  useEffect(() => {
    if (props.kind === "video" && props.frame && vref.current) vref.current.currentTime = props.frame.t;
  }, [props.kind, props.frame, vref]);

  useEffect(() => {
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const c = canvasRef.current;
      const p = propsRef.current;
      const el = p.kind === "video" ? vref.current : imgRef.current;
      if (!c || !el) return;
      const rect = el.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      if (c.width !== Math.round(rect.width * dpr) || c.height !== Math.round(rect.height * dpr)) {
        c.width = Math.round(rect.width * dpr);
        c.height = Math.round(rect.height * dpr);
      }
      const ctx = c.getContext("2d")!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const mw = el instanceof HTMLVideoElement ? el.videoWidth : el.naturalWidth;
      const mh = el instanceof HTMLVideoElement ? el.videoHeight : el.naturalHeight;
      const f = p.frame ?? (el instanceof HTMLVideoElement ? frameAt(p.analysis ?? null, el.currentTime) : p.analysis?.frames[0]);
      drawLandmarks(ctx, f, containBox(rect.width, rect.height, mw, mh), p.layers);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [vref]);

  return (
    <div className={`relative overflow-hidden rounded-xl bg-black ${props.className ?? ""}`}>
      {props.kind === "video" ? (
        <video
          ref={vref}
          src={props.src}
          className="h-full w-full object-contain"
          controls={props.controls}
          muted={props.muted}
          playsInline
          loop
        />
      ) : (
        <img ref={imgRef} src={props.src} className="h-full w-full object-contain" alt="" />
      )}
      <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
      {props.label && (
        <span className="pointer-events-none absolute left-2 top-2 rounded-md bg-black/60 px-2 py-0.5 text-[11px] font-medium text-slate-200">
          {props.label}
        </span>
      )}
    </div>
  );
}

export function LayerToggles({ layers, onChange }: { layers: Layers; onChange: (l: Layers) => void }) {
  const item = (k: keyof Layers, label: string, color: string) => (
    <button
      type="button"
      onClick={() => onChange({ ...layers, [k]: !layers[k] })}
      className={`chip border ${layers[k] ? "border-ink-600 bg-ink-700 text-slate-100" : "border-ink-700 text-ink-400"}`}
    >
      <span className={`h-2 w-2 rounded-full ${layers[k] ? color : "bg-ink-600"}`} />
      {label}
    </button>
  );
  return (
    <div className="flex gap-1.5">
      {item("body", "Body", "bg-accent")}
      {item("hands", "Hands", "bg-cyan-400")}
      {item("face", "Face", "bg-emerald-400")}
    </div>
  );
}
