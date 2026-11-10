import type { GenerationSettings } from "./providers/types.js";

/**
 * Builds the text conditioning sent alongside the image + video.
 * Motion-control models take motion from the video; the prompt only reinforces
 * identity, appearance and scene constraints (it cannot override the model).
 */
export function buildPrompt(s: GenerationSettings, opts: { elementRef?: string }): string {
  const who = opts.elementRef ? `The character ${opts.elementRef} from the reference image` : "The character from the reference image";
  const parts: string[] = [
    `${who} performs exactly the movements of the person in the reference video, with the same timing and speed.`,
  ];
  if (s.identityLock) {
    parts.push(
      "Keep the character's exact face and identity in every frame: same eyes, nose, lips, eyebrows, jawline, facial proportions, skin tone and natural skin texture.",
      "Do not use the reference video person's face or appearance. Same hairstyle, clothing and body proportions as the reference image.",
    );
  }
  if (s.faceDetail) parts.push("Natural blinking, natural facial expressions, consistent teeth and lips, no face distortion.");
  if (s.handDetail) parts.push("Anatomically correct hands with five fingers each, natural wrists and joints.");
  if (s.backgroundPreservation) parts.push("Keep the background from the reference image unchanged and stable.");
  parts.push("Photorealistic, looks like a real camera recording, stable lighting, no flicker.");
  const scene = s.scenePrompt.trim();
  if (scene) parts.push(scene);
  return parts.join(" ");
}
