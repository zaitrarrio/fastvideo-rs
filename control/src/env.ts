// Bindings, secrets and vars of the fv-control Worker (wrangler.toml,
// docs/control/README.md). Secrets are `wrangler secret put`; nothing here
// ever reaches D1, a log line or a response.

export interface Env {
  // --- bindings
  DB: D1Database; // fv-control
  JOBS_DB?: D1Database; // fv-jobs (releases, deployments), read only
  LOGS: R2Bucket; // fv-control-logs
  METRICS?: AnalyticsEngineDataset; // fv_control_metrics
  CLUSTER_OPS: DurableObjectNamespace; // ClusterOps, one per cluster
  ASSETS?: Fetcher; // public/
  /** The edge Worker as a service binding: a Worker cannot fetch another workers.dev Worker of its account (error 1042). */
  EDGE?: Fetcher;

  // --- secrets (never logged, never returned)
  RUNPOD_API_KEY: string;
  CLOUDFLARE_API_KEY?: string; // Analytics Engine SQL
  GITHUB_PAT?: string;
  /** 32+ random bytes, base64: seals cluster secrets and secret env values in D1 (AES-256-GCM). */
  CONTROL_KEK: string;
  /** 32+ random bytes: signs session cookies and CSRF tokens (HMAC-SHA256); also the passphrase pepper. */
  SESSION_SECRET: string;
  /** pbkdf2-sha256$<iter>$<salt b64>$<hash b64> of the owner passphrase (passphrase mode). */
  OWNER_PASSPHRASE_HASH?: string;
  /** The edge Worker's FV_INTERNAL_TOKEN (control_plane = edge: every worker's). */
  EDGE_INTERNAL_TOKEN?: string;
  /** The edge Worker's FV_ADMIN_TOKEN (keys, the families view). */
  EDGE_ADMIN_TOKEN?: string;

  // --- vars
  ENVIRONMENT?: string; // staging | production
  CF_ACCOUNT_ID?: string;
  /** Cloudflare Access: when both are set, every request needs a valid Access JWT. */
  ACCESS_TEAM_DOMAIN?: string; // https://<team>.cloudflareaccess.com
  ACCESS_AUD?: string;
  OWNER_EMAILS?: string; // comma list allowed through Access (empty: any Access identity)
  GITHUB_REPO?: string; // owner/repo
  RELEASE_WORKFLOW?: string; // release.yml
  SERVE_REPO?: string; // ghcr.io/zaitrarrio/fastvideo-rs-serve
  // Upstream bases (tests point them at a mock server).
  RUNPOD_REST?: string;
  RUNPOD_GRAPHQL?: string;
  RUNPOD_HAPI?: string;
  GITHUB_API?: string;
  GHCR?: string;
  CF_API?: string;
  /** `https://{pod}-8000.proxy.runpod.net`; tests: a mock. */
  POD_URL_TEMPLATE?: string;
  /** Public base URL of this Worker (log ingest URL given to pods). */
  PUBLIC_URL?: string;
  /** The account's default balance floor in $ (CLAUDE.md: stop before $8). */
  BALANCE_FLOOR?: string;
  /** control_plane = edge: the edge Worker's public URL (docs/serve/edge-control-plane.md). */
  EDGE_URL?: string;
  /** The edge's D1 database id (its `api_keys` and the workers' job store); FV_D1_DATABASE_ID of edge workers. */
  EDGE_D1_DATABASE_ID?: string;
  /** The edge's outputs bucket (its OUTPUTS binding): FV_R2_BUCKET of edge workers, whose result URLs are presigned there. Unset: edge workers get no R2. */
  EDGE_OUTPUTS_BUCKET?: string;
  /** "1": the cron does nothing (tests drive it by hand). */
  CRON_DISABLED?: string;
}

export type Vars = {
  actor: string; // who: access:<email> | owner | token:<name>
  authKind: "access" | "session" | "token" | "ingest";
  sessionId?: string;
  scope?: "read" | "admin";
};

export const defaults = {
  runpodRest: (e: Env) => e.RUNPOD_REST || "https://rest.runpod.io/v1",
  runpodGraphql: (e: Env) => e.RUNPOD_GRAPHQL || "https://api.runpod.io/graphql",
  runpodHapi: (e: Env) => e.RUNPOD_HAPI || "https://hapi.runpod.net/v1",
  githubApi: (e: Env) => e.GITHUB_API || "https://api.github.com",
  ghcr: (e: Env) => e.GHCR || "https://ghcr.io",
  cfApi: (e: Env) => e.CF_API || "https://api.cloudflare.com/client/v4",
  githubRepo: (e: Env) => e.GITHUB_REPO || "zaitrarrio/fastvideo-rs",
  releaseWorkflow: (e: Env) => e.RELEASE_WORKFLOW || "release.yml",
  serveRepo: (e: Env) => e.SERVE_REPO || "ghcr.io/zaitrarrio/fastvideo-rs-serve",
  podUrl: (e: Env, pod: string) => (e.POD_URL_TEMPLATE || "https://{pod}-8000.proxy.runpod.net").replaceAll("{pod}", pod),
  balanceFloor: (e: Env) => Number(e.BALANCE_FLOOR || "8"),
};
