import { describe, expect, it } from "vitest";
import { DEFAULT_POLICIES } from "../../src/alerts";
import { buildPodVerdict } from "../../src/buildpod";

const pol = DEFAULT_POLICIES;
const H = 3600;

describe("build pod backstop", () => {
  it("defaults: on, 9 h, 15 min past the pod's own idle stop", () => {
    expect([pol.build_pod_backstop, pol.build_pod_max_h, pol.build_pod_idle_grace_min]).toEqual([true, 9, 15]);
  });

  it("stops past the wall-clock backstop, from Runpod's uptime or the pod's", () => {
    expect(buildPodVerdict(9 * H - 1, null, pol)).toBeNull();
    expect(buildPodVerdict(9 * H, null, pol)).toMatch(/up 9\.0 h/);
    expect(buildPodVerdict(null, { uptime_s: 9.2 * H }, pol)).toMatch(/up 9\.2 h/);
    // Even with jobs running: the pod's own grace (8 h + 30 min) is long over.
    expect(buildPodVerdict(9 * H, { idle_s: 0, idle_stop_s: 1200, jobs_active: 2 }, pol)).toMatch(/up/);
  });

  it("stops an idle pod only past its own idle stop plus the grace, and only with no jobs", () => {
    const h = (idle: number, jobs: number) => ({ uptime_s: H, idle_s: idle, idle_stop_s: 1200, jobs_active: jobs });
    expect(buildPodVerdict(H, h(1200 + 15 * 60 - 1, 0), pol)).toBeNull();
    expect(buildPodVerdict(H, h(1200 + 15 * 60, 0), pol)).toMatch(/idle 35 min/);
    expect(buildPodVerdict(H, h(5 * H, 1), pol)).toBeNull();
  });

  it("an old server without timers in /healthz, or no answer: only the uptime rule", () => {
    expect(buildPodVerdict(5 * H, { uptime_s: 5 * H } as any, pol)).toBeNull();
    expect(buildPodVerdict(5 * H, null, pol)).toBeNull();
  });

  it("off switches", () => {
    expect(buildPodVerdict(20 * H, null, { ...pol, build_pod_backstop: false })).toBeNull();
    expect(buildPodVerdict(20 * H, null, { ...pol, build_pod_max_h: 0 })).toBeNull();
  });
});
