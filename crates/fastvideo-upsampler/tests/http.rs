//! The HTTP path against a local fake OpenAI-compatible server: request
//! shape and auth, the hedge, the timeout, non-2xx answers, empty content
//! and bad JSON (`cargo test -p fastvideo-upsampler --features http`).
#![cfg(feature = "http")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fastvideo_upsampler::director::{EntityKind, ExpansionOutput};
use fastvideo_upsampler::http::OpenAiChat;
use fastvideo_upsampler::{
    ChatBackend, Direction, Effort, Entity, PromptFamily, Service, Settings, UpsampleError,
    Upsampler, UpsamplerConfig, WorldBible,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

/// One answer: status, body, delay before answering.
#[derive(Clone)]
struct Answer {
    status: u16,
    body: String,
    delay: Duration,
}

fn ok_content(content: &str) -> Answer {
    Answer {
        status: 200,
        body: json!({"choices": [{"message": {"role": "assistant", "content": content}}]})
            .to_string(),
        delay: Duration::ZERO,
    }
}

struct Seen {
    path: String,
    auth: Option<String>,
    body: Value,
}

/// A fake server: request `n` (from 0) gets `answer(n)`.
struct Fake {
    url: String,
    count: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Seen>>>,
    /// Connections whose answer was never written (the client went away).
    abandoned: Arc<AtomicUsize>,
}

async fn fake(answer: impl Fn(usize) -> Answer + Send + Sync + 'static) -> Fake {
    let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", l.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let abandoned = Arc::new(AtomicUsize::new(0));
    let answer = Arc::new(answer);
    let (c, s, a) = (count.clone(), seen.clone(), abandoned.clone());
    tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = l.accept().await else {
                return;
            };
            let (c, s, a, answer) = (c.clone(), s.clone(), a.clone(), answer.clone());
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut tmp = [0u8; 4096];
                let (head, body) = loop {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..i]).to_string();
                        let len = head
                            .lines()
                            .find_map(|l| {
                                l.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|v| v.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        let mut body = buf[i + 4..].to_vec();
                        while body.len() < len {
                            let n = sock.read(&mut tmp).await.unwrap_or(0);
                            if n == 0 {
                                return;
                            }
                            body.extend_from_slice(&tmp[..n]);
                        }
                        break (head, body);
                    }
                };
                let path = head
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split(' ')
                    .nth(1)
                    .unwrap_or("")
                    .to_owned();
                let auth = head.lines().find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.eq_ignore_ascii_case("authorization")
                        .then(|| v.trim().to_owned())
                });
                let n = c.fetch_add(1, Ordering::SeqCst);
                s.lock().unwrap().push(Seen {
                    path,
                    auth,
                    body: serde_json::from_slice(&body).unwrap_or(Value::Null),
                });
                let ans = answer(n);
                if !ans.delay.is_zero() {
                    // The client aborting shows as a closed socket.
                    let mut probe = [0u8; 1];
                    tokio::select! {
                        _ = tokio::time::sleep(ans.delay) => {}
                        _ = sock.read(&mut probe) => { a.fetch_add(1, Ordering::SeqCst); return; }
                    }
                }
                let reason = if ans.status == 200 { "OK" } else { "Error" };
                let resp = format!(
                    "HTTP/1.1 {} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    ans.status,
                    ans.body.len(),
                    ans.body
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            });
        }
    });
    Fake {
        url,
        count,
        seen,
        abandoned,
    }
}

fn segment() -> ExpansionOutput {
    ExpansionOutput {
        beat_summary: "Mara lights the lamp.".into(),
        entity_ids: vec!["tower".into(), "mara".into()],
        new_entities: vec![Entity {
            id: "gull".into(),
            kind: EntityKind::Prop,
            name: "Gull".into(),
            descriptor: "a grey gull".into(),
        }],
        action: "Mara strikes a match and the lamp flares.".into(),
        camera: "close-up, slow push in".into(),
        lighting: "orange flame on her face".into(),
        audio: "rain on glass".into(),
        music: String::new(),
        dialogue: vec![],
    }
}

fn world() -> WorldBible {
    WorldBible {
        premise: "A lighthouse keeper on a stormy island.".into(),
        style: "35mm film.".into(),
        entities: vec![
            Entity {
                id: "mara".into(),
                kind: EntityKind::Character,
                name: "Mara".into(),
                descriptor: "a wiry woman, yellow oilskin".into(),
            },
            Entity {
                id: "tower".into(),
                kind: EntityKind::Location,
                name: "Lighthouse".into(),
                descriptor: "a white lighthouse on black rocks".into(),
            },
        ],
    }
}

fn upsampler(url: &str, hedge: bool, timeout_ms: u64) -> Upsampler {
    let chat: Arc<dyn ChatBackend> =
        Arc::new(OpenAiChat::new(url, Some("test-key".into()), "test-model").unwrap());
    Upsampler::new(
        UpsamplerConfig {
            memory: 12,
            hedge,
            timeout: Duration::from_millis(timeout_ms),
            ..UpsamplerConfig::default()
        },
        world(),
        chat,
    )
}

#[tokio::test]
async fn next_over_http_sends_the_spec_request() {
    let content = serde_json::to_string(&segment()).unwrap();
    let f = fake(move |_| ok_content(&content)).await;
    let mut u = upsampler(&f.url, false, 5000);
    let s = u
        .next(Some(Direction {
            id: "2".into(),
            text: "The storm breaks.".into(),
        }))
        .await
        .unwrap();
    assert_eq!(
        s.record.prompt,
        "a white lighthouse on black rocks Mara: a wiry woman, yellow oilskin Mara strikes a match and the lamp flares. \
         Camera: close-up, slow push in Lighting: orange flame on her face Sound: rain on glass No spoken words. 35mm film."
    );
    assert_eq!(s.new_entities.len(), 1);
    assert_eq!(f.count.load(Ordering::SeqCst), 1);
    let seen = f.seen.lock().unwrap();
    assert_eq!(seen[0].path, "/v1/chat/completions");
    assert_eq!(seen[0].auth.as_deref(), Some("Bearer test-key"));
    let b = &seen[0].body;
    assert_eq!(b["model"], "test-model");
    assert_eq!(
        b["temperature"].as_f64().map(|t| (t * 10.0).round()),
        Some(8.0)
    );
    assert_eq!(b["messages"][0]["role"], "system");
    assert_eq!(
        b["messages"][0]["content"],
        fastvideo_upsampler::director::system_prompt()
    );
    assert_eq!(b["messages"][1]["role"], "user");
    assert!(b["messages"][1]["content"]
        .as_str()
        .unwrap()
        .ends_with("NEW DIRECTION:\nThe storm breaks."));
    assert_eq!(b["response_format"]["type"], "json_schema");
    assert_eq!(
        b["response_format"]["json_schema"],
        fastvideo_upsampler::director::output_schema()
    );
}

#[tokio::test]
async fn the_hedge_survives_one_failed_attempt() {
    let content = serde_json::to_string(&segment()).unwrap();
    // The first request fails fast, the second succeeds a little later.
    let f = fake(move |n| {
        if n == 0 {
            Answer {
                status: 500,
                body: "{}".into(),
                delay: Duration::ZERO,
            }
        } else {
            Answer {
                delay: Duration::from_millis(50),
                ..ok_content(&content)
            }
        }
    })
    .await;
    let mut u = upsampler(&f.url, true, 5000);
    let s = u.next(None).await.unwrap();
    assert_eq!(s.record.index, 0);
    assert_eq!(f.count.load(Ordering::SeqCst), 2, "two concurrent attempts");
    // Both failing: every error is reported.
    let f = fake(|n| Answer {
        status: 502 + n as u16,
        body: "bad gateway".into(),
        delay: Duration::ZERO,
    })
    .await;
    let e = upsampler(&f.url, true, 5000).next(None).await.unwrap_err();
    let UpsampleError::AllFailed(m) = e else {
        panic!("{e:?}")
    };
    assert!(
        m.contains("upsampler LLM 502: bad gateway")
            && m.contains("upsampler LLM 503: bad gateway"),
        "{m}"
    );
}

#[tokio::test]
async fn the_timeout_aborts_both_attempts() {
    let f = fake(|_| Answer {
        delay: Duration::from_secs(20),
        ..ok_content("{}")
    })
    .await;
    let mut u = upsampler(&f.url, true, 300);
    let t0 = Instant::now();
    assert_eq!(u.next(None).await.unwrap_err(), UpsampleError::Timeout(300));
    assert!(t0.elapsed() < Duration::from_secs(3), "{:?}", t0.elapsed());
    // The server sees both connections go away.
    let deadline = Instant::now() + Duration::from_secs(5);
    while f.abandoned.load(Ordering::SeqCst) < 2 && Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        (
            f.count.load(Ordering::SeqCst),
            f.abandoned.load(Ordering::SeqCst)
        ),
        (2, 2)
    );
    assert_eq!(u.ledger_len(), 0);
}

#[tokio::test]
async fn non_2xx_empty_content_and_bad_json() {
    // Non-2xx: status and (clipped) body, the key never in the message.
    let f = fake(|_| Answer {
        status: 429,
        body: "rate limited".into(),
        delay: Duration::ZERO,
    })
    .await;
    let e = upsampler(&f.url, false, 5000).next(None).await.unwrap_err();
    assert_eq!(
        e,
        UpsampleError::Status {
            status: 429,
            body: "rate limited".into()
        }
    );
    assert!(!e.to_string().contains("test-key"));
    // Empty content and no choices.
    for body in [
        json!({"choices": [{"message": {"content": ""}}]}),
        json!({"choices": []}),
        json!({}),
    ] {
        let b = body.to_string();
        let f = fake(move |_| Answer {
            status: 200,
            body: b.clone(),
            delay: Duration::ZERO,
        })
        .await;
        assert_eq!(
            upsampler(&f.url, false, 5000).next(None).await.unwrap_err(),
            UpsampleError::NoContent,
            "{body}"
        );
    }
    // Content that is not the segment JSON, and a body that is not JSON.
    let f = fake(|_| ok_content("Sure! Here is the next segment.")).await;
    assert!(matches!(
        upsampler(&f.url, false, 5000).next(None).await.unwrap_err(),
        UpsampleError::BadOutput(_)
    ));
    let f = fake(|_| Answer {
        status: 200,
        body: "<html>".into(),
        delay: Duration::ZERO,
    })
    .await;
    assert!(matches!(
        upsampler(&f.url, false, 5000).next(None).await.unwrap_err(),
        UpsampleError::BadOutput(_)
    ));
    // Nothing listening.
    let dead = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", dead.local_addr().unwrap());
    drop(dead);
    assert!(matches!(
        upsampler(&url, false, 5000).next(None).await.unwrap_err(),
        UpsampleError::Transport(_)
    ));
}

#[tokio::test]
async fn service_rewrite_over_http_and_no_key_no_header() {
    let f = fake(|_| {
        ok_content(
            "\"A tabby cat pads across a sunlit kitchen floor, tail high. Slow dolly left.\"",
        )
    })
    .await;
    let s = Service::http(
        &f.url,
        None,
        Settings {
            model: "m".into(),
            hedge: false,
            ..Settings::default()
        },
    )
    .unwrap();
    let p = s
        .rewrite(PromptFamily::Wan, Effort::Balanced, true, "a cat")
        .await
        .unwrap();
    assert_eq!(
        p,
        "A tabby cat pads across a sunlit kitchen floor, tail high. Slow dolly left."
    );
    let seen = f.seen.lock().unwrap();
    assert_eq!(seen[0].auth, None);
    assert!(seen[0].body.get("response_format").is_none());
    assert!(seen[0].body["messages"][0]["content"]
        .as_str()
        .unwrap()
        .contains("image you cannot see"));
    assert_eq!(seen[0].body["messages"][1]["content"], "a cat");
    // Bad settings are refused up front.
    assert!(OpenAiChat::new("ftp://x", None, "m").is_err());
    assert!(OpenAiChat::new("http://x", None, " ").is_err());
    // Debug never prints the key.
    let c = OpenAiChat::new("http://x/v1/", Some("sekret".into()), "m").unwrap();
    assert!(!format!("{c:?}").contains("sekret"));
}

/// A vision request: the user message becomes `[text, image_url...]`
/// content parts (OpenAI's shape, which llama-server `--mmproj` takes).
#[tokio::test]
async fn vision_requests_send_image_url_parts() {
    let f = fake(|_| ok_content("{}")).await;
    let c = OpenAiChat::new(&f.url, None, "qwen3-vl").unwrap();
    let mut req = fastvideo_upsampler::director::request_for(
        "WORLD BIBLE:\nx".into(),
        vec!["data:image/jpeg;base64,AAAA".into()],
    );
    let b = c.body(&req);
    assert_eq!(
        b["messages"][1]["content"],
        json!([{"type": "text", "text": "WORLD BIBLE:\nx"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}])
    );
    // On the wire, as the server receives it.
    c.complete(&req).await.unwrap();
    {
        let seen = f.seen.lock().unwrap();
        let parts = seen[0].body["messages"][1]["content"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(
            parts[1],
            json!({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}})
        );
        assert_eq!(
            seen[0].body["response_format"]["json_schema"]["name"],
            "segment"
        );
    }
    req.images.clear();
    assert_eq!(c.body(&req)["messages"][1]["content"], "WORLD BIBLE:\nx");
}

/// `/tokenize` at the server root (the `/v1` dropped): llama-server's
/// `tokens`, vLLM's `count`; a server without it is asked once.
#[tokio::test]
async fn tokenize_is_feature_checked_once() {
    let f = fake(|n| {
        if n == 0 {
            Answer {
                status: 200,
                body: json!({"tokens": [1, 2, 3]}).to_string(),
                delay: Duration::ZERO,
            }
        } else {
            ok_content("x")
        }
    })
    .await;
    let c = OpenAiChat::new(&f.url, None, "m").unwrap();
    assert_eq!(c.tokenize("a b c").await, Some(3));
    {
        let seen = f.seen.lock().unwrap();
        assert_eq!(seen[0].path, "/tokenize");
        assert_eq!(seen[0].body["content"], "a b c");
    }
    assert_eq!(
        fastvideo_upsampler::http::tokens_of(&json!({"count": 7, "tokens": [1]})),
        Some(7)
    );
    let f = fake(|_| Answer {
        status: 404,
        body: "{}".into(),
        delay: Duration::ZERO,
    })
    .await;
    let c = OpenAiChat::new(&f.url, None, "m").unwrap();
    assert_eq!(c.tokenize("a").await, None);
    assert_eq!(c.tokenize("a").await, None);
    assert_eq!(f.count.load(Ordering::SeqCst), 1, "asked once");
}
