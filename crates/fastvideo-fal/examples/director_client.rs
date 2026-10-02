//! A headless WMA director client for smoke runs on the serving machine
//! (docs/serve/director-causal.md): a str0m peer that opens
//! `{app}/director` on a local fv-serve, configures it, sends prompt
//! updates at set times, stops, and records what arrived.
//!
//! ```text
//! director_client --url http://127.0.0.1:8100 --app fastvideo/longlive \
//!   --seconds 60 --switch-at 15,30,45 --prompts prompts.txt --out /root/dc/run
//! ```
//!
//! `prompts.txt`: one prompt per line; line 1 opens the session, the next
//! lines are sent at the `--switch-at` times (seconds after the first video
//! frame). Writes to `--out`: `events.jsonl` (every control message with its
//! arrival time), `video.h264` (the received access units in order),
//! `audio.opus` (the received Opus packets, Ogg),
//! `frames.tsv` (arrival ms and RTP time per video frame) and
//! `summary.json`. Loopback only: the peer binds `0.0.0.0:0` and talks to
//! the server's host candidates; nothing is exposed.

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_webrtc::channel::ChannelMessage;
use fastvideo_webrtc::host::{AudioLayout, HostConfig, OfferOptions, PeerEvent, RtcHost};
use fastvideo_webrtc::sdp::Direction;
use fastvideo_webrtc::writer::{TrackKind, VideoCodec};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct Args {
    url: String,
    app: String,
    seconds: f64,
    switch_at: Vec<f64>,
    prompts: Vec<String>,
    out: PathBuf,
    key: Option<String>,
    seed: Option<u64>,
    resolution: Option<String>,
}

fn args() -> Result<Args, String> {
    let mut a = Args {
        url: "http://127.0.0.1:8000".into(),
        app: "fastvideo/longlive".into(),
        seconds: 60.0,
        switch_at: vec![15.0, 30.0, 45.0],
        prompts: Vec::new(),
        out: PathBuf::from("director-run"),
        key: std::env::var("FV_KEY").ok().filter(|k| !k.is_empty()),
        seed: None,
        resolution: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().ok_or_else(|| format!("{k} needs a value"));
        match k.as_str() {
            "--url" => a.url = v()?,
            "--app" => a.app = v()?,
            "--seconds" => a.seconds = v()?.parse().map_err(|e| format!("--seconds: {e}"))?,
            "--switch-at" => {
                a.switch_at = v()?.split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse::<f64>().map_err(|e| format!("--switch-at: {e}"))).collect::<Result<_, _>>()?
            }
            "--prompts" => {
                let p = v()?;
                let text = std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?;
                a.prompts = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_owned).collect();
            }
            "--prompt" => a.prompts.push(v()?),
            "--out" => a.out = PathBuf::from(v()?),
            "--resolution" => a.resolution = Some(v()?),
            "--seed" => a.seed = Some(v()?.parse().map_err(|e| format!("--seed: {e}"))?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if a.prompts.is_empty() {
        return Err("give --prompts FILE or --prompt TEXT".into());
    }
    Ok(a)
}

/// One HTTP/1.1 POST with a JSON body to a plain-HTTP local server.
async fn post(url: &str, path: &str, key: Option<&str>, body: &Value) -> Result<(u16, Value), String> {
    let hostport = url.trim_start_matches("http://").trim_end_matches('/');
    let mut s = tokio::net::TcpStream::connect(hostport).await.map_err(|e| format!("connect {hostport}: {e}"))?;
    let body = body.to_string();
    let auth = key.map(|k| format!("authorization: Key {k}\r\n")).unwrap_or_default();
    let req = format!(
        "POST {path} HTTP/1.1\r\nhost: {hostport}\r\ncontent-type: application/json\r\n{auth}content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.map_err(|e| e.to_string())?;
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).await.map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&buf);
    let (head, rest) = text.split_once("\r\n\r\n").ok_or("malformed response")?;
    let status: u16 = head.split_whitespace().nth(1).and_then(|s| s.parse().ok()).ok_or("no status")?;
    let body = if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        let mut out = String::new();
        let mut r = rest;
        while let Some((len, tail)) = r.split_once("\r\n") {
            let n = usize::from_str_radix(len.trim(), 16).unwrap_or(0);
            if n == 0 || tail.len() < n {
                break;
            }
            out.push_str(&tail[..n]);
            r = tail[n..].trim_start_matches("\r\n");
        }
        out
    } else {
        rest.to_owned()
    };
    Ok((status, serde_json::from_str(&body).unwrap_or(Value::String(body))))
}

/// A minimal Ogg Opus writer (RFC 7845) for the received stereo packets,
/// one packet per page; granule positions from the 48 kHz RTP times.
struct OggOpus {
    f: std::fs::File,
    seq: u32,
    first_rtp: Option<u64>,
}

fn ogg_crc(data: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &b in data {
        crc ^= u32::from(b) << 24;
        for _ in 0..8 {
            crc = if crc & 0x8000_0000 != 0 { (crc << 1) ^ 0x04c1_1db7 } else { crc << 1 };
        }
    }
    crc
}

impl OggOpus {
    fn create(path: &std::path::Path) -> std::io::Result<Self> {
        let mut me = Self { f: std::fs::File::create(path)?, seq: 0, first_rtp: None };
        let mut head = b"OpusHead".to_vec();
        head.extend_from_slice(&[1, 2]);
        head.extend_from_slice(&312u16.to_le_bytes());
        head.extend_from_slice(&48_000u32.to_le_bytes());
        head.extend_from_slice(&0i16.to_le_bytes());
        head.push(0);
        me.page(&head, 0, 0x02)?;
        let mut tags = b"OpusTags".to_vec();
        let vendor = b"fv-director-client";
        tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        tags.extend_from_slice(vendor);
        tags.extend_from_slice(&0u32.to_le_bytes());
        me.page(&tags, 0, 0)?;
        Ok(me)
    }

    fn page(&mut self, packet: &[u8], granule: u64, flags: u8) -> std::io::Result<()> {
        let mut segs = Vec::new();
        let mut n = packet.len();
        loop {
            let s = n.min(255);
            segs.push(s as u8);
            n -= s;
            if s < 255 {
                break;
            }
        }
        let mut p = b"OggS".to_vec();
        p.push(0);
        p.push(flags);
        p.extend_from_slice(&granule.to_le_bytes());
        p.extend_from_slice(&0x6676_6463u32.to_le_bytes());
        p.extend_from_slice(&self.seq.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.push(segs.len() as u8);
        p.extend_from_slice(&segs);
        p.extend_from_slice(packet);
        let crc = ogg_crc(&p);
        p[22..26].copy_from_slice(&crc.to_le_bytes());
        self.seq += 1;
        self.f.write_all(&p)
    }

    fn packet(&mut self, data: &[u8], rtp: u64) {
        let first = *self.first_rtp.get_or_insert(rtp);
        // 20 ms packets: the granule is the end of this packet (+312 pre-skip).
        let granule = rtp.saturating_sub(first) + 960 + 312;
        let _ = self.page(data, granule, 0);
    }
}

#[derive(Default)]
struct Media {
    /// (ms since start, rtp) per video frame.
    frames: Vec<(f64, u64)>,
    keyframes: u64,
    audio_packets: u64,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let a = match args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("director_client: {e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(a).await {
        eprintln!("director_client: {e}");
        std::process::exit(1);
    }
}

async fn run(a: Args) -> Result<(), String> {
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let now_ms = move || t0.elapsed().as_secs_f64() * 1000.0;
    let host = RtcHost::bind(HostConfig { udp_bind: Some("0.0.0.0:0".parse().unwrap()), ice_servers: Vec::new(), ..HostConfig::default() })
        .await
        .map_err(|e| format!("bind: {e}"))?;
    let (pending, offer) = host
        .offer(OfferOptions {
            video: Some(Direction::RecvOnly),
            audio: Some((Direction::RecvOnly, AudioLayout::Stereo)),
            channels: vec!["control".into()],
            video_codecs: vec![VideoCodec::H264],
            ..Default::default()
        })
        .await
        .map_err(|e| format!("offer: {e}"))?;
    let app_id = format!("{}/director", a.app);
    let (st, v) = post(&a.url, "/wma/session", a.key.as_deref(), &json!({"app_id": app_id, "sdp": offer, "type": "offer"})).await?;
    if st != 200 {
        return Err(format!("/wma/session: {st} {v}"));
    }
    let answer = v["sdp"].as_str().ok_or("no sdp in the answer")?.to_owned();
    let session_id = v["session_id"].as_str().unwrap_or_default().to_owned();
    eprintln!("session {session_id}");
    let peer = pending.accept_answer(&answer).await.map_err(|e| format!("answer: {e}"))?;
    let (handle, mut events) = peer.split();

    let media = Arc::new(Mutex::new(Media::default()));
    let (msg_tx, mut msgs) = tokio::sync::mpsc::unbounded_channel::<(f64, Value)>();
    let mut h264 = std::fs::File::create(a.out.join("video.h264")).map_err(|e| e.to_string())?;
    let mut ogg = OggOpus::create(&a.out.join("audio.opus")).map_err(|e| e.to_string())?;
    let m2 = media.clone();
    let reader = tokio::spawn(async move {
        while let Some(e) = events.recv().await {
            match e {
                PeerEvent::Message(m) => {
                    if let Some(Ok(v)) = m.as_text().map(serde_json::from_str::<Value>) {
                        let _ = msg_tx.send((now_ms(), v));
                    }
                }
                PeerEvent::Media { kind: TrackKind::Video, rtp_time, keyframe, data, .. } => {
                    let _ = h264.write_all(&data);
                    let mut l = m2.lock().unwrap();
                    l.frames.push((now_ms(), rtp_time));
                    l.keyframes += u64::from(keyframe);
                }
                PeerEvent::Media { kind: TrackKind::Audio, rtp_time, data, .. } => {
                    ogg.packet(&data, rtp_time);
                    m2.lock().unwrap().audio_packets += 1;
                }
                PeerEvent::Closed(_) => break,
                _ => {}
            }
        }
    });
    // Heartbeats every 5 s.
    let (url, key, sid) = (a.url.clone(), a.key.clone(), session_id.clone());
    let beats = tokio::spawn(async move {
        let mut t = tokio::time::interval(Duration::from_secs(5));
        loop {
            t.tick().await;
            let _ = post(&url, "/wma/session/heartbeat", key.as_deref(), &json!({"session_id": sid})).await;
        }
    });

    let mut log = std::fs::File::create(a.out.join("events.jsonl")).map_err(|e| e.to_string())?;
    let mut all: Vec<(f64, Value)> = Vec::new();
    let mut record = |t: f64, v: &Value, all: &mut Vec<(f64, Value)>| {
        let _ = writeln!(log, "{}", json!({"t_ms": (t * 10.0).round() / 10.0, "msg": v}));
        all.push((t, v.clone()));
    };
    // session_info, then configure.
    let deadline = Instant::now() + Duration::from_secs(120);
    let info = loop {
        let (t, m) = tokio::time::timeout_at(deadline.into(), msgs.recv()).await.map_err(|_| "no session_info")?.ok_or("closed")?;
        record(t, &m, &mut all);
        if m["type"] == "session_info" {
            break m;
        }
    };
    eprintln!("session_info: fps {} chunk {} s, context {} frames, causal {}", info["fps"], info["chunk_seconds"], info["continuation_context_frames"], info["causal"]);
    let mut cfg = json!({"type": "configure", "prompt_version": 1, "prompt": a.prompts[0]});
    if let Some(s) = a.seed {
        cfg["seed"] = s.into();
    }
    if let Some(r) = &a.resolution {
        cfg["resolution"] = r.clone().into();
    }
    let t_configure = now_ms();
    handle.send_message(ChannelMessage::text("control", cfg.to_string())).await.map_err(|e| e.to_string())?;

    // Run: switches at their times after the first video frame.
    let mut sent: Vec<(u64, f64, String)> = Vec::new();
    let mut next = 0usize;
    let mut first_video: Option<f64> = None;
    let mut stopped = false;
    let mut ended = false;
    let hard_end = Instant::now() + Duration::from_secs_f64(a.seconds + 240.0);
    while !ended && Instant::now() < hard_end {
        if first_video.is_none() {
            first_video = media.lock().unwrap().frames.first().map(|f| f.0);
        }
        if let Some(fv) = first_video {
            let since = (now_ms() - fv) / 1000.0;
            if next < a.switch_at.len() && next + 1 < a.prompts.len() && since >= a.switch_at[next] {
                let v = next as u64 + 2;
                let text = a.prompts[next + 1].clone();
                let p = json!({"type": "prompt", "prompt_version": v, "prompt": text});
                handle.send_message(ChannelMessage::text("control", p.to_string())).await.map_err(|e| e.to_string())?;
                eprintln!("{:.1} s: prompt v{v}", since);
                sent.push((v, now_ms(), text));
                next += 1;
            }
            if !stopped && since >= a.seconds {
                handle.send_message(ChannelMessage::text("control", json!({"type": "stop"}).to_string())).await.map_err(|e| e.to_string())?;
                stopped = true;
            }
        }
        match tokio::time::timeout(Duration::from_millis(100), msgs.recv()).await {
            Ok(Some((t, m))) => {
                record(t, &m, &mut all);
                match m["type"].as_str() {
                    Some("chunk") => eprintln!(
                        "chunk {} v{} frames {} gen {:.2}s pace {} fps buffer {:.2}s recaches {}",
                        m["chunk_index"], m["prompt_version"], m["generated_frame_count"], m["generation_seconds"].as_f64().unwrap_or(0.0),
                        m["causal"]["generation_fps"], m["buffer_depth_seconds"].as_f64().unwrap_or(0.0), m["causal"]["recaches"]
                    ),
                    Some("error") | Some("deadline_missed") | Some("prompt_applied") | Some("prompt_rejected") => eprintln!("{m}"),
                    Some("session_metrics") if m["final"] == true => ended = true,
                    _ => {}
                }
            }
            Ok(None) => ended = true,
            Err(_) => {}
        }
    }
    beats.abort();
    handle.close();
    let _ = tokio::time::timeout(Duration::from_secs(2), reader).await;

    // Summary.
    let (frames, keyframes, audio_packets) = {
        let l = media.lock().unwrap();
        (l.frames.clone(), l.keyframes, l.audio_packets)
    };
    let frames = &frames;
    let fv = frames.first().map(|f| f.0);
    let gaps: Vec<f64> = frames.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let max_gap = gaps.iter().copied().fold(0.0, f64::max);
    let freezes = gaps.iter().filter(|g| **g > 250.0).count();
    let span_s = match (frames.first(), frames.last()) {
        (Some(a), Some(b)) if frames.len() > 1 => (b.0 - a.0) / 1000.0,
        _ => 0.0,
    };
    let rtp_span = match (frames.first(), frames.last()) {
        (Some(a), Some(b)) if frames.len() > 1 => (b.1 - a.1) as f64 / 90_000.0,
        _ => 0.0,
    };
    let mut tsv = String::from("t_ms\trtp\n");
    for (t, r) in frames {
        tsv.push_str(&format!("{t:.1}\t{r}\n"));
    }
    let _ = std::fs::write(a.out.join("frames.tsv"), tsv);
    let of = |ty: &str| all.iter().filter(|(_, m)| m["type"] == ty).cloned().collect::<Vec<_>>();
    let chunks = of("chunk");
    let applied = of("prompt_applied");
    let switches: Vec<Value> = sent
        .iter()
        .map(|(v, t, text)| {
            let ap = applied.iter().find(|(_, m)| m["prompt_version"] == *v).map(|(ta, _)| ta - t);
            let first_chunk = chunks.iter().find(|(_, m)| m["prompt_version"].as_u64().is_some_and(|x| x >= *v)).map(|(_, m)| m["chunk_index"].clone());
            json!({"prompt_version": v, "sent_s": (t - fv.unwrap_or(0.0)) / 1000.0, "applied_after_ms": ap, "first_chunk": first_chunk, "prompt": text})
        })
        .collect();
    let recaches: u64 = chunks.iter().map(|(_, m)| m["causal"]["recaches"].as_u64().unwrap_or(0)).sum();
    let gen_fps: Vec<f64> = chunks.iter().filter_map(|(_, m)| m["causal"]["generation_fps"].as_f64()).collect();
    let gen_s: f64 = chunks.iter().filter_map(|(_, m)| m["generation_seconds"].as_f64()).sum();
    let play_s: f64 = chunks.iter().filter_map(|(_, m)| m["playback_seconds"].as_f64()).sum();
    let fin = of("session_metrics").into_iter().rev().find(|(_, m)| m["final"] == true).map(|x| x.1);
    let summary = json!({
        "app": a.app,
        "session_id": session_id,
        "session_info": info,
        "ttff_ms": fv.map(|f| f - t_configure),
        "video_frames": frames.len(),
        "keyframes": keyframes,
        "audio_packets": audio_packets,
        "received_span_s": span_s,
        "received_fps": if span_s > 0.0 { (frames.len() - 1) as f64 / span_s } else { 0.0 },
        "rtp_fps": if rtp_span > 0.0 { (frames.len() - 1) as f64 / rtp_span } else { 0.0 },
        "max_frame_gap_ms": max_gap,
        "freezes_over_250ms": freezes,
        "chunks": chunks.len(),
        "deadline_missed": of("deadline_missed").len(),
        "errors": of("error").into_iter().map(|x| x.1).collect::<Vec<_>>(),
        "recaches_reported": recaches,
        "generation_fps_min": gen_fps.iter().copied().fold(f64::INFINITY, f64::min),
        "generation_fps_mean": if gen_fps.is_empty() { 0.0 } else { gen_fps.iter().sum::<f64>() / gen_fps.len() as f64 },
        "generation_over_playback": if play_s > 0.0 { gen_s / play_s } else { 0.0 },
        "switches": switches,
        "final_metrics": fin,
    });
    let text = serde_json::to_string_pretty(&summary).unwrap_or_default();
    std::fs::write(a.out.join("summary.json"), &text).map_err(|e| e.to_string())?;
    println!("{text}");
    host.shutdown().await;
    Ok(())
}
