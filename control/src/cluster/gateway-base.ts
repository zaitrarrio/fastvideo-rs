// Generated from configs/serve/gateway-pods.toml (everything before the first [[pools]]).
// test/unit/payloads.test.ts fails when the two drift apart.
export const GATEWAY_BASE_PODS = `[server]
bind = "0.0.0.0:8000"
state_dir = "/fvstate"
shutdown_grace_s = 10
sync_timeout_s = 600

[auth]
mode = "keys"

[artifacts]
backend = "auto"
url_ttl_s = 86400

[jobs]
backend = "auto"
progress_interval_ms = 1000
stale_after_s = 900

[engine]
backend = "remote"

[gateway]
caps_refresh_s = 60
tick_s = 5
metrics_window_s = 600
watch_poll_ms = 1000
runpod_api_base = "https://api.runpod.ai/v2"
reactor_model = "fasth3"
inline_inputs_max_bytes = 8388608  # inputs up to 8 MiB per job ride in the dispatch (no R2 hop; serverless: ≤ 6 MiB)
input_passthrough = true           # large video/audio given as a public URL: the worker fetches it
stage_inputs_for_retry = true      # copy inputs to R2 after the dispatch, for a re-dispatch

[aliases]
"MiniMax-H3" = "fasth3"
"MiniMax-H3-Turbo" = "fasth3"
"MiniMax-H3-Max" = "sol-h3"

[protocols]
openai_videos = true
fastwan = false
minimax = true
fal = true
fal_director = true
ltx = true
reactor = true
native = true
# Every pool's fal apps (the workers mount the same ids).
fal_apps = ["minimax/h3-max", "minimax/h3-turbo", "fastvideo/ltx-turbo", "lightricks/ltx-2.5", "fal-ai/wan"]

[limits]
queue_max = 256
body_max_mb = 64

`;
