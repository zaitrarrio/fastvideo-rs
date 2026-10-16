// fv-app public model catalog: what the gallery and playground show.
// Curated (not live engine data): the portal is public and must not depend
// on operator infrastructure. Sample videos are placeholders until per-model
// release clips are produced (spec open question #7).
export interface CatalogModel {
  id: string;
  title: string;
  tagline: string;
  description: string;
  family: string;
  tier: string | null;
  tasks: string[];
  /** Public model id for the playground's /v1/videos call. */
  api_model: string;
  sample_video: string | null;
  poster: string | null;
}

export const CATALOG: CatalogModel[] = [
  {
    id: "fasth3",
    title: "H3 Turbo",
    tagline: "The fastest way to conjure video",
    description:
      "MiniMax H3 turbo tier: 4-step generation with native audio. Text to video in seconds, with dialogue, sound effects, and music rendered alongside the picture.",
    family: "h3",
    tier: "Turbo",
    tasks: ["text-to-video", "image-to-video"],
    api_model: "fasth3",
    sample_video: null,
    poster: null,
  },
  {
    id: "h3-max",
    title: "H3 Max",
    tagline: "Maximum quality, still fast",
    description:
      "MiniMax H3 max tier: the highest-quality configuration that passes the quality gate. For hero shots where every detail matters.",
    family: "h3",
    tier: "Max",
    tasks: ["text-to-video", "image-to-video"],
    api_model: "h3-max",
    sample_video: null,
    poster: null,
  },
  {
    id: "wan22-ti2v-5b",
    title: "Wan 2.2 5B",
    tagline: "Open weights, turbo speed",
    description:
      "Wan 2.2 TI2V-5B: FastWan turbo and the 50-step max tier. Strong motion coherence at an approachable price.",
    family: "wan",
    tier: "Turbo",
    tasks: ["text-to-video", "image-to-video"],
    api_model: "wan22-ti2v-5b",
    sample_video: null,
    poster: null,
  },
  {
    id: "ltx-2.5",
    title: "LTX 2.5",
    tagline: "Lightning text to video",
    description:
      "LTX-2.5 turbo tier: fal-grade text-to-video distilled for speed. Ideal for rapid iteration and previews.",
    family: "ltx2",
    tier: "Turbo",
    tasks: ["text-to-video"],
    api_model: "ltx-2.5",
    sample_video: null,
    poster: null,
  },
];
