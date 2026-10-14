//! `stream_outputs::hls::read_audio` against a local HTTP server (no
//! network): playlists in Cloudflare's shape (the audio a separate
//! `#EXT-X-MEDIA TYPE=AUDIO` rendition, the video variants without audio) and
//! in the muxed shape (audio in the variant's own segments), with segments
//! that ffmpeg encodes here: a tone must read as audible, silence must not.
//! Skips when ffmpeg is missing.

#![cfg(feature = "http-client")]

use std::path::{Path as FsPath, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::Router;
use fastvideo_serve::stream_outputs::hls::read_audio;

fn ffmpeg(args: &[&str]) -> bool {
    Command::new(fastvideo_media::tools::ffmpeg_bin())
        .args(["-v", "error", "-nostdin", "-y"])
        .args(args)
        .status()
        .is_ok_and(|s| s.success())
}

/// `<dir>/<name>.m3u8` + `<name>N.ts`: 3 s MPEG-TS segments of `inputs`.
fn hls(dir: &FsPath, name: &str, inputs: &[&str], codecs: &[&str]) -> bool {
    let mut a: Vec<String> = inputs.iter().map(|s| s.to_string()).collect();
    a.extend(codecs.iter().map(|s| s.to_string()));
    let seg = dir.join(format!("{name}%d.ts"));
    let pl = dir.join(format!("{name}.m3u8"));
    a.extend(
        [
            "-f",
            "hls",
            "-hls_time",
            "3",
            "-hls_list_size",
            "0",
            "-hls_segment_filename",
        ]
        .map(String::from),
    );
    a.push(seg.display().to_string());
    a.push(pl.display().to_string());
    let r: Vec<&str> = a.iter().map(String::as_str).collect();
    ffmpeg(&r)
}

async fn serve(dir: PathBuf) -> String {
    async fn file(
        State(dir): State<PathBuf>,
        Path(name): Path<String>,
    ) -> Result<Vec<u8>, StatusCode> {
        if name.contains("..") || name.contains('/') {
            return Err(StatusCode::BAD_REQUEST);
        }
        std::fs::read(dir.join(name)).map_err(|_| StatusCode::NOT_FOUND)
    }
    let app = Router::new().route("/{name}", get(file)).with_state(dir);
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = l.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(l, app).await.ok() });
    format!("http://{addr}")
}

#[tokio::test]
async fn reads_a_separate_rendition_and_muxed_audio() {
    let d = tempfile::tempdir().unwrap();
    let dir = d.path();
    let tone = [
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=9:sample_rate=48000",
    ];
    let quiet = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "9"];
    let video = [
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x180:rate=24:duration=9",
    ];
    let aac = ["-c:a", "aac", "-b:a", "96k", "-ac", "2"];
    let ok = hls(dir, "audio", &tone, &aac)
        && hls(dir, "silent", &quiet, &aac)
        && hls(
            dir,
            "video",
            &video,
            &["-c:v", "mpeg2video", "-g", "24", "-an"],
        )
        && hls(
            dir,
            "muxed",
            &[&video[..], &tone[..]].concat(),
            &["-c:v", "mpeg2video", "-g", "24", "-c:a", "aac", "-ac", "2"],
        );
    if !ok {
        eprintln!("skipping: ffmpeg could not make the HLS fixtures");
        return;
    }
    // Cloudflare's shape: the variant names the audio group; its segments have no audio.
    let cf = |audio: &str| {
        format!(
            "#EXTM3U\n#EXT-X-VERSION:6\n#EXT-X-INDEPENDENT-SEGMENTS\n\
             #EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID=\"group_audio\",NAME=\"original\",DEFAULT=YES,AUTOSELECT=YES,URI=\"{audio}.m3u8\"\n\
             #EXT-X-STREAM-INF:RESOLUTION=320x180,CODECS=\"avc1.64001f,mp4a.40.2\",BANDWIDTH=900000,AUDIO=\"group_audio\"\n\
             video.m3u8\n"
        )
    };
    std::fs::write(dir.join("cf.m3u8"), cf("audio")).unwrap();
    std::fs::write(dir.join("cf-silent.m3u8"), cf("silent")).unwrap();
    std::fs::write(
        dir.join("mux.m3u8"),
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=900000,CODECS=\"mp2v,mp4a.40.2\"\nmuxed.m3u8\n\
         #EXT-X-STREAM-INF:BANDWIDTH=200000,CODECS=\"mp2v,mp4a.40.2\"\nmuxed.m3u8\n",
    )
    .unwrap();
    let base = serve(dir.to_owned()).await;
    let http = reqwest::Client::new();
    let deadline = || Instant::now() + Duration::from_secs(30);
    let out = dir.join("read.seg");

    let a = read_audio(
        &http,
        &format!("{base}/cf.m3u8"),
        Duration::from_secs(60),
        deadline(),
        &out,
    )
    .await
    .unwrap();
    println!("rendition: {}", serde_json::to_string(&a).unwrap());
    assert_eq!(a.source, "rendition");
    assert_eq!(a.group.as_deref(), Some("group_audio"));
    assert_eq!(a.playlist_uri, "audio.m3u8");
    assert_eq!(a.first_sequence, 0);
    assert_eq!(a.segments, a.first_listed);
    assert_eq!(a.codec.as_deref(), Some("aac"));
    let l = a.levels.as_ref().unwrap();
    // lavfi's sine is 1/8 of full scale (peak -18 dBFS, RMS -21 dBFS); the
    // mono -> stereo -> mono round trip takes 3 dB off: -24 / -21.
    assert!(
        l.audible_seconds >= 8
            && (l.rms_dbfs - -24.1).abs() < 2.0
            && (l.peak_dbfs - -21.1).abs() < 2.0,
        "{l:?}"
    );
    assert!(a.audible);
    assert!(a.master_text.contains("TYPE=AUDIO") && a.media_text.contains("#EXTINF"));

    // A window shorter than the playlist stops early.
    let a = read_audio(
        &http,
        &format!("{base}/cf.m3u8"),
        Duration::from_secs(3),
        deadline(),
        &out,
    )
    .await
    .unwrap();
    assert_eq!(a.segments, 1);

    let s = read_audio(
        &http,
        &format!("{base}/cf-silent.m3u8"),
        Duration::from_secs(60),
        deadline(),
        &out,
    )
    .await
    .unwrap();
    println!("silent rendition: {}", serde_json::to_string(&s).unwrap());
    assert_eq!(s.source, "rendition");
    assert_eq!(s.codec.as_deref(), Some("aac"));
    assert!(!s.audible, "{s:?}");

    // The video variant alone (what the readers used to decode) has no audio.
    let v = read_audio(
        &http,
        &format!("{base}/video.m3u8"),
        Duration::from_secs(60),
        deadline(),
        &out,
    )
    .await
    .unwrap();
    assert_eq!(v.source, "muxed");
    assert_eq!(v.codec, None);
    assert!(!v.audible);

    let m = read_audio(
        &http,
        &format!("{base}/mux.m3u8"),
        Duration::from_secs(60),
        deadline(),
        &out,
    )
    .await
    .unwrap();
    println!("muxed: {}", serde_json::to_string(&m).unwrap());
    assert_eq!(m.source, "muxed");
    assert_eq!(m.playlist_uri, "muxed.m3u8");
    assert!(
        m.variant
            .as_deref()
            .is_some_and(|v| v.contains("BANDWIDTH=200000")),
        "the lightest variant: {:?}",
        m.variant
    );
    assert_eq!(m.codec.as_deref(), Some("aac"));
    assert!(m.audible);

    // A master whose audio group is not listed yet never resolves before the deadline.
    std::fs::write(
        dir.join("pending.m3u8"),
        "#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1,AUDIO=\"group_audio\"\nvideo.m3u8\n",
    )
    .unwrap();
    let e = read_audio(
        &http,
        &format!("{base}/pending.m3u8"),
        Duration::from_secs(9),
        Instant::now() + Duration::from_secs(4),
        &out,
    )
    .await
    .unwrap_err();
    assert!(e.contains("group_audio"), "{e}");
}
