//! Avatar mode: Reactor's `ltx` model contract (a photo and a script in,
//! speech and video out) over the engine's windowed
//! [`AvatarPlayer`](fastvideo_engine_service::AvatarPlayer).
//!
//! - Conditions: `set_avatar_image` (an upload, decoded and re-encoded as
//!   PNG), `set_script`, `set_prompt` (scene and delivery), `set_wpm`
//!   (80-220, default 140), `set_duration_seconds` (0 derives it from the
//!   script), `set_seed`; extension `set_voice_audio` (an uploaded speech
//!   file: the windows become audio-to-video). Each setter replies with its
//!   `*_accepted` message and broadcasts `state_update`. A change made while
//!   a take runs applies to the next take and is listed in
//!   `queued_changes`.
//! - `start` plans the take ([`plan`]) and hands it to the player;
//!   `pause` / `resume` / `stop` / `reset` follow Reactor's rules
//!   (`command_error` otherwise). The player's events become
//!   `generation_started`, `window_progress`, `generation_*`, plus the
//!   extensions `window_built` (build time, real-time factor) and
//!   `window_started` (latency, stall).
//! - Take length: 4-300 s (Reactor); the session as a whole is capped by
//!   `/start_session` `max_seconds` or `[reactor] avatar_session_max_s`
//!   (design §5.2).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use fastvideo_engine_service::stream::avatar::{
    derived_seconds, effective_seconds, plan, word_count, AvatarConfig, AvatarEvent, AvatarPlayer, AvatarStatus,
    AvatarTake, PlanInput, WPM_DEFAULT, WPM_MAX, WPM_MIN,
};
use fastvideo_engine_service::{
    spawn_clip_pacer, ClipPacerConfig, ClipSession, IdlePolicy, PacedStream, TickStart,
};
use fastvideo_protocol::{ApiError, FrameGrid};
use serde_json::{json, Map, Value};

use crate::driver::{Driver, Outbox, Outcome};
use crate::uploads::{Uploads, RESOLVE_TIMEOUT};
use crate::wire::ServerMsg;

/// Avatar-mode settings (`[reactor] avatar_*`).
#[derive(Clone, Debug, PartialEq)]
pub struct AvatarSettings {
    /// Longest window (seconds of speech per generation).
    pub window_s: f64,
    /// Delivered canvas (Reactor `ltx`: 640x352).
    pub size: (u32, u32),
    /// Session ceiling in video seconds (`/start_session` `max_seconds` may
    /// ask for less).
    pub session_max_s: u32,
    /// Built windows allowed ahead of playout.
    pub lookahead: usize,
    /// Playout speed; tests only.
    pub speed: f64,
}

impl Default for AvatarSettings {
    fn default() -> Self {
        Self { window_s: 10.0, size: (640, 352), session_max_s: 1800, lookahead: 2, speed: 1.0 }
    }
}

#[derive(Clone, Debug, Default)]
struct Image {
    path: PathBuf,
    name: String,
    width: u32,
    height: u32,
}

#[derive(Clone, Debug, Default)]
struct Voice {
    path: PathBuf,
    name: String,
    seconds: f64,
}

#[derive(Clone, Debug)]
struct Conds {
    image: Option<Image>,
    voice: Option<Voice>,
    script: String,
    prompt: String,
    wpm: u32,
    duration: f64,
    seed: u64,
    queued: Vec<String>,
}

impl Conds {
    fn new(seed: u64) -> Self {
        Self { image: None, voice: None, script: String::new(), prompt: String::new(), wpm: WPM_DEFAULT, duration: 0.0, seed, queued: Vec::new() }
    }

    fn derived(&self) -> f64 {
        match &self.voice {
            Some(v) => (v.seconds * 10.0).round() / 10.0,
            None => derived_seconds(word_count(&self.script), self.wpm),
        }
    }

    fn effective(&self) -> f64 {
        effective_seconds(self.derived(), self.duration)
    }

    fn startable(&self) -> bool {
        self.image.is_some() && (!self.script.trim().is_empty() || self.voice.is_some())
    }
}

struct Inner {
    player: AvatarPlayer,
    out: Outbox,
    uploads: Arc<Uploads>,
    conds: Mutex<Conds>,
    session_seed: u64,
    dir: PathBuf,
    fps: u32,
    grid: FrameGrid,
    window_s: f64,
}

/// The avatar-mode [`crate::driver::Driver`].
pub struct AvatarDriver {
    inner: Arc<Inner>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

impl AvatarDriver {
    /// Starts the player (no audience yet) and the clip pacer (idle: hold
    /// the last frame, ticks from the start so the audio clock runs).
    pub fn start(
        session: ClipSession,
        settings: &AvatarSettings,
        uploads: Arc<Uploads>,
        seed: u64,
        out: Outbox,
    ) -> Result<(Self, PacedStream), ApiError> {
        let spec = session.spec().clone();
        let grid = session.caps().frames.clone();
        let dir = std::env::temp_dir().join("fv-sessions").join(format!("{}-avatar-inputs", session.id()));
        let (player, outputs) = AvatarPlayer::start(
            session,
            AvatarConfig {
                delivered: Some(settings.size),
                lookahead: settings.lookahead,
                speed: settings.speed,
                ..AvatarConfig::default()
            },
        )?;
        let paced = spawn_clip_pacer(
            outputs.media,
            ClipPacerConfig {
                idle: IdlePolicy::Hold,
                start: TickStart::Immediately,
                canvas: settings.size,
                speed: settings.speed,
                ..ClipPacerConfig::for_spec(&spec)
            },
        )?;
        let inner = Arc::new(Inner {
            player,
            out,
            uploads,
            conds: Mutex::new(Conds::new(seed)),
            session_seed: seed,
            dir,
            fps: spec.fps,
            grid,
            window_s: settings.window_s,
        });
        let mut events = outputs.events;
        let weak = Arc::downgrade(&inner);
        tokio::spawn(async move {
            while let Some(ev) = events.recv().await {
                let Some(inner) = weak.upgrade() else { break };
                inner.forward(ev).await;
            }
        });
        Ok((Self { inner }, paced))
    }

    pub fn player(&self) -> &AvatarPlayer {
        &self.inner.player
    }
}

impl Inner {
    fn state(&self, st: &AvatarStatus) -> Value {
        let c = lock(&self.conds);
        let generating = st.generating;
        let mut valid = vec!["set_avatar_image", "set_script", "set_prompt", "set_wpm", "set_duration_seconds", "set_seed", "set_voice_audio", "get_state", "reset"];
        if c.startable() && !generating {
            valid.push("start");
        }
        if generating && !st.paused {
            valid.push("pause");
        }
        if generating && st.paused {
            valid.push("resume");
        }
        if generating {
            valid.push("stop");
        }
        json!({
            "script": c.script, "prompt": c.prompt,
            "has_avatar_image": c.image.is_some(), "has_voice_audio": c.voice.is_some(),
            "wpm": c.wpm, "wpm_min": WPM_MIN, "wpm_max": WPM_MAX,
            "duration_seconds": c.duration, "effective_seconds": c.effective(), "seed": c.seed,
            "ready": c.startable() && !generating, "generating": generating, "paused": st.paused, "finished": st.finished,
            "valid_commands": valid, "queued_changes": c.queued,
            "window_index": st.window_index, "total_windows": st.total_windows, "seconds_sent": st.seconds_sent,
            "windows_built": st.windows_built, "stalls": st.stalls, "stalled_seconds": st.stalled_s,
            "first_frame_seconds": st.first_frame_s,
            "avatar_image": c.image.as_ref().map(|i| json!({"name": i.name, "width": i.width, "height": i.height})),
            "voice_audio": c.voice.as_ref().map(|v| json!({"name": v.name, "seconds": r2(v.seconds)})),
        })
    }

    fn state_now(&self) -> Value {
        self.state(&self.player.status())
    }

    fn broadcast_state(&self) {
        self.out.broadcast(ServerMsg::broadcast("state_update", self.state_now()));
    }

    fn refuse(&self, command: &str, reason: impl Into<String>) -> Outcome {
        let reason = reason.into();
        tracing::info!(command, reason = %reason, "avatar command refused");
        self.out.broadcast(ServerMsg::broadcast("command_error", json!({"command": command, "reason": reason})));
        Outcome::Ack
    }

    /// Notes a change made while a take runs.
    fn queue_change(&self, c: &mut Conds, what: &str) {
        if self.player.status().generating && !c.queued.iter().any(|q| q == what) {
            c.queued.push(what.to_owned());
        }
    }

    async fn forward(&self, ev: AvatarEvent) {
        let (kind, data) = match ev {
            AvatarEvent::Started { seconds, total_windows, width, height } => (
                "generation_started",
                json!({"seconds": seconds, "total_windows": total_windows, "width": width, "height": height}),
            ),
            AvatarEvent::WindowBuilt(r) => (
                "window_built",
                json!({"window_index": r.index, "total_windows": r.total_windows, "kind": r.kind, "frames": r.frames,
                       "seconds": r.seconds, "build_seconds": r.build_s, "rtf": r.rtf}),
            ),
            AvatarEvent::WindowStarted { index, total_windows, seconds_sent, since_start_s, stalled_s } => (
                "window_started",
                json!({"window_index": index, "total_windows": total_windows, "seconds_sent": seconds_sent,
                       "since_start_seconds": since_start_s, "stalled_seconds": stalled_s}),
            ),
            AvatarEvent::WindowProgress { index, total_windows, seconds_sent, total_seconds } => (
                "window_progress",
                json!({"window_index": index, "total_windows": total_windows, "seconds_sent": seconds_sent, "total_seconds": total_seconds}),
            ),
            AvatarEvent::Paused { seconds_sent } => ("generation_paused", json!({"seconds_sent": seconds_sent})),
            AvatarEvent::Resumed { seconds_sent } => ("generation_resumed", json!({"seconds_sent": seconds_sent})),
            AvatarEvent::Stopped { seconds_sent } => ("generation_stopped", json!({"seconds_sent": seconds_sent})),
            AvatarEvent::Complete { seconds_sent } => ("generation_complete", json!({"seconds_sent": seconds_sent})),
            AvatarEvent::Failed { reason, seconds_sent } => {
                ("generation_failed", json!({"reason": reason, "seconds_sent": seconds_sent}))
            }
        };
        self.out.broadcast(ServerMsg::broadcast(kind, data));
        // The player's status catches up right after the event.
        tokio::time::sleep(Duration::from_millis(5)).await;
        self.broadcast_state();
    }

    async fn upload(&self, v: &Value) -> Result<crate::uploads::Upload, Outcome> {
        let id = v.get("upload_id").and_then(Value::as_str).unwrap_or_default();
        self.uploads.resolve(id, RESOLVE_TIMEOUT).await.ok_or_else(|| Outcome::Error {
            code: "unresolved_upload".into(),
            message: format!("upload `{id}` was not received"),
        })
    }

    async fn command(&self, name: &str, a: Map<String, Value>) -> Outcome {
        let reply = |kind: &str, data: Value| Outcome::Reply(kind.to_owned(), data);
        match name {
            "get_state" => reply("state_update", self.state_now()),
            "set_avatar_image" => {
                let up = match self.upload(&a["avatar_image"]).await {
                    Ok(u) => u,
                    Err(o) => return o,
                };
                let dst = self.dir.join(format!("avatar-{}.png", uuid::Uuid::new_v4().simple()));
                let (src, d2) = (up.path.clone(), dst.clone());
                let decoded = tokio::task::spawn_blocking(move || normalize_image(&src, &d2)).await;
                let (width, height) = match decoded {
                    Ok(Ok(wh)) => wh,
                    Ok(Err(e)) => return self.refuse(name, e),
                    Err(e) => return self.refuse(name, format!("decoding the image: {e}")),
                };
                {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "avatar_image");
                    if let Some(old) = c.image.replace(Image { path: dst, name: up.name.clone(), width, height }) {
                        let _ = std::fs::remove_file(old.path);
                    }
                }
                self.broadcast_state();
                reply("avatar_image_accepted", json!({"name": up.name, "width": width, "height": height}))
            }
            "set_voice_audio" => {
                let up = match self.upload(&a["voice_audio"]).await {
                    Ok(u) => u,
                    Err(o) => return o,
                };
                let src = up.path.clone();
                let dst = self.dir.join(format!("voice-{}", uuid::Uuid::new_v4().simple()));
                let d2 = dst.clone();
                let probed = tokio::task::spawn_blocking(move || {
                    std::fs::create_dir_all(d2.parent().unwrap_or(Path::new(".")))
                        .and_then(|()| std::fs::copy(&src, &d2))
                        .map_err(|e| format!("storing the voice: {e}"))?;
                    audio_seconds(&d2)
                })
                .await;
                let seconds = match probed {
                    Ok(Ok(s)) if s >= 1.0 => s,
                    Ok(Ok(s)) => return self.refuse(name, format!("the voice audio is {s:.2} s; at least 1 s")),
                    Ok(Err(e)) => return self.refuse(name, e),
                    Err(e) => return self.refuse(name, format!("probing the voice: {e}")),
                };
                let effective = {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "voice_audio");
                    if let Some(old) = c.voice.replace(Voice { path: dst, name: up.name.clone(), seconds }) {
                        let _ = std::fs::remove_file(old.path);
                    }
                    c.effective()
                };
                self.broadcast_state();
                reply("voice_audio_accepted", json!({"name": up.name, "seconds": r2(seconds), "effective_seconds": effective}))
            }
            "set_script" => {
                let script = a["script"].as_str().unwrap_or_default().to_owned();
                let data = {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "script");
                    c.script = script;
                    json!({"words": word_count(&c.script), "derived_seconds": derived_seconds(word_count(&c.script), c.wpm), "effective_seconds": c.effective()})
                };
                self.broadcast_state();
                reply("script_accepted", data)
            }
            "set_prompt" => {
                let prompt = a["prompt"].as_str().unwrap_or_default().to_owned();
                {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "prompt");
                    c.prompt = prompt.clone();
                }
                self.broadcast_state();
                reply("prompt_accepted", json!({"prompt": prompt}))
            }
            "set_wpm" => {
                let wpm = a["wpm"].as_u64().unwrap_or(u64::from(WPM_DEFAULT)) as u32;
                let data = {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "wpm");
                    c.wpm = wpm;
                    json!({"wpm": wpm, "derived_seconds": c.derived(), "effective_seconds": c.effective()})
                };
                self.broadcast_state();
                reply("wpm_accepted", data)
            }
            "set_duration_seconds" => {
                let d = a["duration_seconds"].as_f64().unwrap_or(0.0);
                if d > 0.0 && d < fastvideo_engine_service::stream::avatar::TAKE_MIN_S {
                    // Reactor clamps a short duration up to 4 s.
                    tracing::debug!(d, "duration clamped to the 4 s minimum");
                }
                let data = {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "duration_seconds");
                    c.duration = d;
                    json!({"duration_seconds": d, "effective_seconds": c.effective()})
                };
                self.broadcast_state();
                reply("duration_accepted", data)
            }
            "set_seed" => {
                let seed = a["seed"].as_u64().unwrap_or(0);
                {
                    let mut c = lock(&self.conds);
                    self.queue_change(&mut c, "seed");
                    c.seed = seed;
                }
                self.broadcast_state();
                reply("seed_accepted", json!({"seed": seed}))
            }
            "start" => {
                let (take, plan_len) = {
                    let mut c = lock(&self.conds);
                    let Some(image) = c.image.clone() else {
                        return self.refuse(name, "set_avatar_image first");
                    };
                    if c.script.trim().is_empty() && c.voice.is_none() {
                        return self.refuse(name, "set_script first");
                    }
                    let p = plan(&PlanInput {
                        script: &c.script,
                        scene: &c.prompt,
                        wpm: c.wpm,
                        duration_s: c.duration,
                        window_s: self.window_s,
                        fps: self.fps,
                        grid: self.grid.clone(),
                        voice_s: c.voice.as_ref().map(|v| v.seconds),
                    });
                    c.queued.clear();
                    let n = p.windows.len();
                    (AvatarTake { plan: p, image: image.path, seed: c.seed, voice: c.voice.as_ref().map(|v| v.path.clone()) }, n)
                };
                tracing::info!(windows = plan_len, seconds = take.plan.effective_seconds, "avatar start");
                match self.player.start_take(take).await {
                    Ok(()) => Outcome::Ack,
                    Err(e) => self.refuse(name, e),
                }
            }
            "pause" | "resume" | "stop" => {
                let r = match name {
                    "pause" => self.player.pause().await,
                    "resume" => self.player.resume().await,
                    _ => self.player.stop().await,
                };
                match r {
                    Ok(()) => Outcome::Ack,
                    Err(e) => self.refuse(name, e),
                }
            }
            "reset" => {
                let was_generating = self.player.status().generating;
                if was_generating {
                    let _ = self.player.stop().await;
                }
                {
                    let mut c = lock(&self.conds);
                    if let Some(i) = c.image.take() {
                        let _ = std::fs::remove_file(i.path);
                    }
                    if let Some(v) = c.voice.take() {
                        let _ = std::fs::remove_file(v.path);
                    }
                    *c = Conds::new(self.session_seed);
                }
                self.out.broadcast(ServerMsg::broadcast("generation_reset", json!({"was_generating": was_generating})));
                self.broadcast_state();
                Outcome::Ack
            }
            _ => Outcome::Error { code: "invalid_command".into(), message: format!("unknown command `{name}`") },
        }
    }
}

/// Decodes an uploaded image (PNG or JPEG, by content) and writes it as a
/// PNG; returns its size.
fn normalize_image(src: &Path, dst: &Path) -> Result<(u32, u32), String> {
    let img = image::ImageReader::open(src)
        .and_then(|r| r.with_guessed_format())
        .map_err(|e| format!("reading the image: {e}"))?
        .decode()
        .map_err(|e| format!("the avatar image is not a PNG or JPEG image: {e}"))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    if w < 64 || h < 64 {
        return Err(format!("the avatar image is {w}x{h}; at least 64x64"));
    }
    if let Some(d) = dst.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("storing the image: {e}"))?;
    }
    img.save(dst).map_err(|e| format!("storing the image: {e}"))?;
    Ok((w, h))
}

/// Duration of an audio file (ffprobe `format=duration`).
fn audio_seconds(path: &Path) -> Result<f64, String> {
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration", "-of", "csv=p=0"])
        .arg(path)
        .output()
        .map_err(|e| format!("ffprobe not available: {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout);
    match text.trim().parse::<f64>() {
        Ok(s) if out.status.success() && s.is_finite() => Ok(s),
        _ => Err("the voice audio is not a readable audio file".into()),
    }
}

#[async_trait]
impl Driver for AvatarDriver {
    async fn command(&self, _conn: u32, name: &str, args: Map<String, Value>) -> Outcome {
        self.inner.command(name, args).await
    }

    fn greet(&self, conn: u32) {
        self.inner.out.to(conn, ServerMsg::broadcast("state_update", self.inner.state_now()));
    }

    fn peers_changed(&self, connected: usize) {
        let p = self.inner.player.clone();
        tokio::spawn(async move { p.set_audience(connected > 0).await });
    }

    async fn close(&self) {
        tracing::info!("avatar driver closing");
        self.inner.player.close().await;
        tracing::info!("avatar player closed");
        let _ = std::fs::remove_dir_all(&self.inner.dir);
    }
}
