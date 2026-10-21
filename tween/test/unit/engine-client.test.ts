import { describe, expect, it } from "vitest";
import { EngineClient, EngineError, buildRequest, parseJob } from "../../src/engine-client";

const FIRST = "data:image/png;base64,AAAA";
const LAST = "data:image/png;base64,BBBB";

describe("buildRequest", () => {
  it("sends first and last panels as two image references for a keyframe shot", () => {
    const body = buildRequest({
      family: "ltx2",
      model: "ltx25-distill-sol",
      mode: "keyframes",
      prompt: "the door swings open",
      seconds: 6,
      seed: 42,
      aspect_ratio: "16:9",
      images: [FIRST, LAST],
    });
    expect(body).toEqual({
      model: "ltx25-distill-sol",
      prompt: "the door swings open",
      seconds: 6,
      seed: 42,
      aspect_ratio: "16:9",
      image_reference: [{ image_url: FIRST }, { image_url: LAST }],
    });
    expect(body).not.toHaveProperty("task");
  });

  it("names ref2va on H3 for a reference shot", () => {
    const body = buildRequest({
      family: "h3",
      model: "h3-ref2v-turbo",
      mode: "reference",
      prompt: "she turns",
      seconds: 5,
      seed: 1,
      aspect_ratio: "9:16",
      images: [FIRST, LAST, FIRST],
    });
    expect(body.task).toBe("ref2va");
    expect(body.image_reference).toHaveLength(3);
  });

  it("refuses a keyframe shot without exactly two images and a reference shot on LTX", () => {
    const base = { family: "ltx2" as const, model: "m", prompt: "p", seconds: 6, seed: 1, aspect_ratio: "16:9" };
    expect(() => buildRequest({ ...base, mode: "keyframes", images: [FIRST] })).toThrow(/two/);
    expect(() => buildRequest({ ...base, mode: "reference", images: [FIRST, LAST] })).toThrow(/H3/);
  });

  it("sends a blank prompt as a single space rather than an empty field", () => {
    const body = buildRequest({
      family: "h3",
      model: "m",
      mode: "keyframes",
      prompt: "   ",
      seconds: 5,
      seed: 1,
      aspect_ratio: "16:9",
      images: [FIRST, LAST],
    });
    expect(body.prompt).toBe(" ");
  });
});

describe("parseJob", () => {
  it("reads the /v1/videos job view", () => {
    const job = parseJob({
      id: "video_1",
      object: "video",
      status: "completed",
      progress: 100,
      seconds: "6",
      size: "1280x768",
      created_at: 1000,
      completed_at: 1004,
      inference_time_s: 2.5,
      error: null,
    });
    expect(job).toEqual({
      id: "video_1",
      status: "completed",
      progress: 100,
      seconds: 6,
      size: "1280x768",
      created_at_s: 1000,
      completed_at_s: 1004,
      inference_time_s: 2.5,
      error: null,
    });
  });

  it("keeps the engine's failure message and tolerates missing metrics", () => {
    const job = parseJob({
      id: "video_2",
      status: "failed",
      progress: 30,
      seconds: "5",
      created_at: 1,
      error: { code: "generation_failed", message: "out of memory" },
    });
    expect(job.status).toBe("failed");
    expect(job.error).toBe("out of memory");
    expect(job.inference_time_s).toBeNull();
    expect(job.completed_at_s).toBeNull();
  });

  it("rejects a body that is not a job", () => {
    expect(() => parseJob({ status: "completed" })).toThrow();
    expect(() => parseJob({ id: "x", status: "weird" })).toThrow();
  });
});

type Call = { url: string; init: RequestInit };
function fakeFetch(respond: (c: Call) => Response) {
  const calls: Call[] = [];
  const f = async (input: RequestInfo | URL, init?: RequestInit) => {
    const c = { url: String(input), init: init ?? {} };
    calls.push(c);
    return respond(c);
  };
  return { f: f as typeof fetch, calls };
}

describe("EngineClient", () => {
  it("submits with a bearer key and parses the queued job", async () => {
    const { f, calls } = fakeFetch(
      () => new Response(JSON.stringify({ id: "video_9", status: "queued", progress: 0, seconds: "6", created_at: 5 }), { status: 200 }),
    );
    const client = new EngineClient("http://engine:8000/", "sk-test", f);
    const job = await client.submit({ model: "m", prompt: "p", seconds: 6 });
    expect(job.id).toBe("video_9");
    expect(calls[0]!.url).toBe("http://engine:8000/v1/videos");
    expect(calls[0]!.init.method).toBe("POST");
    expect(new Headers(calls[0]!.init.headers).get("authorization")).toBe("Bearer sk-test");
  });

  it("turns an OpenAI error envelope into an EngineError with the engine's message", async () => {
    const { f } = fakeFetch(
      () =>
        new Response(JSON.stringify({ error: { message: "model `x` does not support task keyframes", param: "task" } }), {
          status: 400,
        }),
    );
    const client = new EngineClient("http://engine", "k", f);
    const err = await client.submit({}).catch((e) => e);
    expect(err).toBeInstanceOf(EngineError);
    expect(err.status).toBe(400);
    expect(err.message).toContain("does not support task keyframes");
  });

  it("calls fetch unbound, as workerd requires of the global fetch", async () => {
    // workerd throws "Illegal invocation" when fetch runs with a foreign `this`.
    const strict = function (this: unknown) {
      if (this !== undefined && this !== globalThis) throw new TypeError("Illegal invocation");
      return Promise.resolve(Response.json({ id: "video_1", status: "queued", progress: 0, seconds: 6, created_at: 1 }));
    } as unknown as typeof fetch;
    const job = await new EngineClient("http://engine", "k", strict).get("video_1");
    expect(job.id).toBe("video_1");
  });

  it("reports an unreachable engine as a 502 EngineError", async () => {
    const f = (async () => {
      throw new TypeError("connect ECONNREFUSED");
    }) as unknown as typeof fetch;
    const err = await new EngineClient("http://engine", "k", f).get("video_1").catch((e) => e);
    expect(err).toBeInstanceOf(EngineError);
    expect(err.status).toBe(502);
  });

  it("fetches a job and its content by id, escaping the id", async () => {
    const { f, calls } = fakeFetch((c) =>
      c.url.endsWith("/content")
        ? new Response(new Uint8Array([0, 0, 0, 24]), { status: 200, headers: { "content-type": "video/mp4" } })
        : new Response(JSON.stringify({ id: "a/b", status: "in_progress", progress: 40, seconds: 6, created_at: 1 })),
    );
    const client = new EngineClient("http://engine", "k", f);
    expect((await client.get("a/b")).progress).toBe(40);
    const body = await client.content("a/b");
    expect(new Uint8Array(await new Response(body.stream).arrayBuffer())).toHaveLength(4);
    expect(body.contentType).toBe("video/mp4");
    expect(calls.map((c) => c.url)).toEqual(["http://engine/v1/videos/a%2Fb", "http://engine/v1/videos/a%2Fb/content"]);
  });
});
