// Owner's reference implementation of the director prompt upsampler (2026-10-13),
// kept verbatim as the spec. fv-serve implements it in Rust (see
// docs/serve/prompt-upsampling.md); this file is not built.
// src/upsampler.ts
export type EntityKind = "character" | "location" | "prop";

export interface Entity {
  id: string;
  kind: EntityKind;
  name: string;
  descriptor: string; // canonical appearance text, spliced verbatim every segment
}

export interface WorldBible {
  premise: string;
  style: string; // appended to every segment
  entities: Entity[];
}

export interface Direction {
  id: string;
  text: string;
}

export interface SegmentRecord {
  index: number;
  directionId: string | null;
  summary: string;
  prompt: string;
}

export interface ExpandedSegment extends SegmentRecord {
  newEntities: Entity[];
}

interface DialogueLine {
  speaker_id: string;
  line: string;
}

interface ExpansionOutput {
  beat_summary: string;
  entity_ids: string[];
  new_entities: Entity[];
  action: string;
  camera: string;
  lighting: string;
  audio: string;
  dialogue: DialogueLine[];
}

export interface UpsamplerConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
  memory: number; // prior segments kept as context (Director default 12, max 50)
  maxPromptChars: number; // keep under the text encoder limit (~1800 chars for umT5 512 tokens)
  hedge: boolean;
  timeoutMs: number;
}

const SYSTEM_PROMPT = `You are the showrunner for a continuous, never-ending live video stream.
Each turn you write the NEXT segment of one unbroken take. You receive the world bible,
the registry of established entities, the recent segment history, and an optional new direction.

Rules:
- Continue directly from the last segment. No recaps, no time skips unless the direction demands one.
- If a direction is given, make it happen in this segment, staying consistent with established people and places.
- If no direction is given, advance the story by one natural beat.
- Refer to existing people, places, and props ONLY by their entity ids. Never describe their appearance.
- Introduce a new entity only if the story needs it; give it a short id and a concrete visual descriptor.
- "action" is present-tense, physical and visible: what moves, where, how. One continuous shot.
- "audio" describes ambience, foley and score. Spoken words go only in "dialogue", verbatim.
- Write concrete visual language. No meta commentary, no references to cameras "filming a show".`;

const OUTPUT_SCHEMA = {
  name: "segment",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    required: [
      "beat_summary",
      "entity_ids",
      "new_entities",
      "action",
      "camera",
      "lighting",
      "audio",
      "dialogue",
    ],
    properties: {
      beat_summary: { type: "string" },
      entity_ids: { type: "array", items: { type: "string" } },
      new_entities: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["id", "kind", "name", "descriptor"],
          properties: {
            id: { type: "string" },
            kind: { type: "string", enum: ["character", "location", "prop"] },
            name: { type: "string" },
            descriptor: { type: "string" },
          },
        },
      },
      action: { type: "string" },
      camera: { type: "string" },
      lighting: { type: "string" },
      audio: { type: "string" },
      dialogue: {
        type: "array",
        items: {
          type: "object",
          additionalProperties: false,
          required: ["speaker_id", "line"],
          properties: {
            speaker_id: { type: "string" },
            line: { type: "string" },
          },
        },
      },
    },
  },
} as const;

function buildUserMessage(
  world: WorldBible,
  registry: Map<string, Entity>,
  history: SegmentRecord[],
  direction: Direction | null,
): string {
  const entities = [...registry.values()]
    .map((e) => `- ${e.id} (${e.kind}, "${e.name}")`)
    .join("\n");
  const recent = history
    .map((s) => `[${s.index}] ${s.summary}\n${s.prompt}`)
    .join("\n\n");
  return [
    `WORLD BIBLE:\n${world.premise}`,
    `ESTABLISHED ENTITIES:\n${entities || "(none)"}`,
    `RECENT SEGMENTS (oldest first):\n${recent || "(stream is starting)"}`,
    direction
      ? `NEW DIRECTION:\n${direction.text}`
      : "NEW DIRECTION:\n(none, advance the story one beat)",
  ].join("\n\n");
}

async function callModel(
  cfg: UpsamplerConfig,
  user: string,
  signal: AbortSignal,
): Promise<ExpansionOutput> {
  const res = await fetch(`${cfg.baseUrl}/chat/completions`, {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      authorization: `Bearer ${cfg.apiKey}`,
    },
    body: JSON.stringify({
      model: cfg.model,
      temperature: 0.8,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: user },
      ],
      response_format: { type: "json_schema", json_schema: OUTPUT_SCHEMA },
    }),
  });
  if (!res.ok) throw new Error(`upsampler LLM ${res.status}: ${await res.text()}`);
  const body = (await res.json()) as {
    choices: { message: { content: string } }[];
  };
  const content = body.choices[0]?.message.content;
  if (!content) throw new Error("upsampler LLM returned no content");
  return JSON.parse(content) as ExpansionOutput;
}

async function expandWithHedge(
  cfg: UpsamplerConfig,
  user: string,
): Promise<ExpansionOutput> {
  const attempts = cfg.hedge ? 2 : 1;
  const controllers = Array.from({ length: attempts }, () => new AbortController());
  const timer = setTimeout(() => controllers.forEach((c) => c.abort()), cfg.timeoutMs);
  try {
    return await Promise.any(controllers.map((c) => callModel(cfg, user, c.signal)));
  } finally {
    clearTimeout(timer);
    controllers.forEach((c) => c.abort());
  }
}

function assemblePrompt(
  out: ExpansionOutput,
  registry: Map<string, Entity>,
  world: WorldBible,
  maxChars: number,
): string {
  const cast = out.entity_ids
    .map((id) => registry.get(id))
    .filter((e): e is Entity => e !== undefined);
  const location = cast.find((e) => e.kind === "location");
  const others = cast.filter((e) => e.kind !== "location");
  const nameOf = (id: string) => registry.get(id)?.name ?? "a voice";

  const parts = [
    location ? location.descriptor : "",
    ...others.map((e) => `${e.name}: ${e.descriptor}`),
    out.action,
    ...out.dialogue.map((d) => `${nameOf(d.speaker_id)} says, "${d.line}"`),
    `Camera: ${out.camera}`,
    `Lighting: ${out.lighting}`,
    `Sound: ${out.audio}${out.dialogue.length === 0 ? " No spoken words." : ""}`,
    world.style,
  ].filter((p) => p.trim().length > 0);

  let prompt = parts.join(" ");
  if (prompt.length > maxChars) {
    // Style is the most expendable; identity text and action are not.
    prompt = parts.slice(0, -1).join(" ").slice(0, maxChars);
  }
  return prompt;
}

export class Upsampler {
  private readonly registry = new Map<string, Entity>();
  private readonly ledger: SegmentRecord[] = [];

  constructor(
    private readonly cfg: UpsamplerConfig,
    private readonly world: WorldBible,
  ) {
    for (const e of world.entities) this.registry.set(e.id, e);
  }

  async next(direction: Direction | null): Promise<ExpandedSegment> {
    const history = this.ledger.slice(-this.cfg.memory);
    const user = buildUserMessage(this.world, this.registry, history, direction);
    const out = await expandWithHedge(this.cfg, user);

    const accepted = out.new_entities.filter((e) => !this.registry.has(e.id));
    for (const e of accepted) this.registry.set(e.id, e);
    out.entity_ids = out.entity_ids.filter((id) => this.registry.has(id));

    const record: SegmentRecord = {
      index: this.ledger.length,
      directionId: direction?.id ?? null,
      summary: out.beat_summary,
      prompt: assemblePrompt(out, this.registry, this.world, this.cfg.maxPromptChars),
    };
    this.ledger.push(record);
    return { ...record, newEntities: accepted };
  }

  snapshot(): { registry: Entity[]; ledger: SegmentRecord[] } {
    return { registry: [...this.registry.values()], ledger: [...this.ledger] };
  }
}
