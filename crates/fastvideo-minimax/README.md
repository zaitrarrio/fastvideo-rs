# fastvideo-minimax

fv-serve adapter: MiniMax V2 video generation (create, query, list, delete, callbacks). See [docs/serve/design.md](../../docs/serve/design.md) §4.3 (WP-07) and [research-minimax-fastvideo.md](../../docs/serve/research-minimax-fastvideo.md) §1.2-1.7.

| Route | Reply |
|---|---|
| `POST /v2/video_generation` | 200 `{"task_id": "<18 digits>"}` |
| `GET /v2/query/video_generation/{task_id}` | `{"task": VideoTask}` |
| `GET /v2/query/video_generation` | `{"items": [VideoTask], "total"}` |
| `DELETE /v2/video_generation/{task_id}` | `{"task_id", "action", "status"}` |
| `POST /v2/h3_context_ir`, `POST /v2/video_regeneration` | 400 (not served) |

Mounting (the binary):

```rust
let mm = Arc::new(MiniMax::new(MiniMaxConfig::default()));
let ctx = ServeCtx::builder(cfg, engine)
    .renderer(ProtocolId::MiniMaxV2, fastvideo_minimax::callback_renderer(&mm))
    .build().await?;
let app = mm.router().merge(ctx.routes()).with_state(ctx);
```

Models: `MiniMax-H3` (4–15 s; 768P, 2K → 400 gap), `MiniMax-H3-Max` (5–15 s; 480P/768P),
and our tiers `MiniMax-H3-Turbo` / `MiniMax-H3-Draft` (4–15 s; 480P/768P). Resolution
order: `minimax.models`, the server alias, the `h3-max`/`h3-turbo`/`h3-draft` tier
alias, the tier tag. Rate limits: 300 creates/min and 30 in-flight tasks per key.
