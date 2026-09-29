//! WebRTC signalling under `/sessions/{sid}/transport/webrtc` (reactor
//! §3.3, design §5.7): HTTP only, no WebSocket.
//!
//! | Route | Behaviour |
//! |---|---|
//! | `GET ice_servers` | `{"ice_servers":[{"uris":[…],"credentials"?}]}` |
//! | `POST connections` | **201** `{connection_id: 1002..9999, track_map}` (model perspective) |
//! | `POST`/`PUT connections/{cid}/sdp_params` | **202** `{connection_id}`; the answer is produced in the background. PUT is a re-offer (reconnect) on the same id |
//! | `GET connections/{cid}/sdp_params` | **202** while negotiating, then **200** `{sdp_answer, connection_id}` **once** (taken) |
//! | `POST connections/{cid}/ice_candidates` | **202**; buffered before the offer (≤256 per connection), an empty candidate is end-of-candidates |
//!
//! Every route first answers **400** `No session running` outside
//! WAITING/STREAMING/ORPHANED and **404** `Unknown session` for another id.
//! Answers are non-trickle: every server candidate plus
//! `a=end-of-candidates`; `a=x-reactor-frame-metadata` is never mirrored.
//! Offers past `max_connections` (64) get 503, re-offers are always admitted,
//! and a connection must be up within 30 s of its offer (the host's
//! negotiation deadline).

use std::collections::HashMap;
use std::sync::Arc;

use fastvideo_webrtc::channel::ChannelPolicy;
use fastvideo_webrtc::host::{AnswerOptions, AudioLayout};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::gateway;
use crate::session::{lock, Answer, Conn, Live, Reactor, Refusal, SESSION_ID};
use crate::wire::WireVersion;

/// `track_mapping[]` of an offer (client perspective).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct TrackMapping {
    #[serde(default)]
    pub mid: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub direction: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct IceCredentials {
    pub ufrag: String,
    pub pwd: String,
}

/// `SdpParamsRequest` (unknown fields such as `client_info` are ignored).
#[derive(Clone, Debug, Deserialize)]
pub struct SdpParams {
    pub sdp_offer: String,
    #[serde(default)]
    pub track_mapping: Vec<TrackMapping>,
    #[serde(default)]
    pub ice_servers: Option<Vec<Value>>,
    #[serde(default)]
    pub ice_credentials: Option<IceCredentials>,
    #[serde(default)]
    pub port_range: Option<Value>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Candidate {
    #[serde(default)]
    pub candidate: String,
    #[serde(default)]
    pub sdp_mid: Option<String>,
    #[serde(default)]
    pub sdp_mline_index: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Candidates {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub is_final: bool,
}

fn ice_chars_ok(s: &str, min: usize) -> bool {
    (min..=256).contains(&s.len()) && s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'+' || b == b'/')
}

/// What `GET …/sdp_params` answers.
#[derive(Clone, Debug, PartialEq)]
pub enum AnswerPoll {
    Pending,
    Ready(String),
}

impl Reactor {
    /// `require_session_running` plus the session-id check.
    pub(crate) fn require_running(&self, sid: &str) -> Result<Arc<Live>, Refusal> {
        if !self.state().is_running() {
            return Err(Refusal::new(400, "No session running"));
        }
        if sid != SESSION_ID {
            return Err(Refusal::new(404, "Unknown session"));
        }
        self.live().ok_or_else(|| Refusal::new(400, "No session running"))
    }

    /// `GET ice_servers`.
    pub fn ice_servers(&self, sid: &str) -> Result<Value, Refusal> {
        self.require_running(sid)?;
        let servers: Vec<Value> = self.host().ice_servers().iter().map(|s| s.to_reactor_json()).collect();
        Ok(json!({"ice_servers": servers}))
    }

    /// `POST connections`: a random unused id in 1002..=9999.
    pub fn register_connection(&self, sid: &str) -> Result<Value, Refusal> {
        let live = self.require_running(sid)?;
        let mut conns = lock(&live.conns);
        if conns.len() >= self.config().max_registered {
            return Err(Refusal::new(503, "No connection ids left"));
        }
        let mut id = 0u32;
        for _ in 0..10_000 {
            let r = 1002 + (uuid::Uuid::new_v4().as_u128() % 8998) as u32;
            if !conns.contains_key(&r) {
                id = r;
                break;
            }
        }
        if id == 0 {
            return Err(Refusal::new(503, "No connection ids left"));
        }
        conns.insert(
            id,
            Conn { gen: 0, answer: Answer::None, candidates: Vec::new(), peer: None, mapping: HashMap::new() },
        );
        Ok(json!({"connection_id": id, "track_map": Reactor::track_map(&live.tracks)}))
    }

    /// `POST`/`PUT sdp_params`: starts answering `params.sdp_offer`.
    pub fn offer(&self, sid: &str, cid: u32, params: SdpParams, header: Option<&str>) -> Result<Value, Refusal> {
        let live = self.require_running(sid)?;
        if params.sdp_offer.trim().is_empty() {
            return Err(Refusal::new(422, "sdp_offer is required"));
        }
        if let Some(c) = &params.ice_credentials {
            if !ice_chars_ok(&c.ufrag, 4) || !ice_chars_ok(&c.pwd, 22) {
                return Err(Refusal::new(422, "ice_credentials: ufrag 4..256 and pwd 22..256 ice-chars"));
            }
        }
        if params.port_range.is_some() {
            // One shared mux socket serves every peer (design §5.8).
            tracing::debug!(cid, "port_range ignored: the WebRTC host uses one shared socket");
        }
        let seed = WireVersion::from_header(header);
        let (gen, old) = {
            let mut conns = lock(&live.conns);
            let active = conns
                .iter()
                .filter(|(id, c)| **id != cid && (c.peer.is_some() || c.answer == Answer::Pending))
                .count();
            let c = conns.get_mut(&cid).ok_or_else(|| Refusal::new(404, "Unknown connection"))?;
            let reoffer = c.gen > 0;
            if !reoffer && active >= self.config().max_connections {
                return Err(Refusal::new(503, "Connection limit reached"));
            }
            c.gen += 1;
            c.answer = Answer::Pending;
            c.mapping = params
                .track_mapping
                .iter()
                .filter(|t| !t.name.is_empty() && !t.mid.is_empty())
                .map(|t| (t.name.clone(), t.mid.clone()))
                .collect();
            (c.gen, c.peer.take())
        };
        if let Some(p) = old {
            // A re-offer replaces the wire; the session stays up.
            p.close();
        }
        let opts = AnswerOptions {
            video: true,
            video_codecs: live.codecs.clone(),
            audio: live.tracks.has_audio().then_some(AudioLayout::Mono),
            channels: ChannelPolicy::reactor(),
            start_paused: true,
            ice_credentials: params.ice_credentials.map(|c| (c.ufrag, c.pwd)),
            ..AnswerOptions::default()
        };
        let me = self.clone();
        let offer = params.sdp_offer;
        tokio::spawn(async move {
            let res = me.host().answer(&offer, opts).await;
            let current = me.is_current(&live);
            let (peer, buffered, mapping) = {
                let mut conns = lock(&live.conns);
                let Some(c) = conns.get_mut(&cid).filter(|c| c.gen == gen && current) else {
                    // Superseded by a newer offer, or the session ended.
                    if let Ok((p, _)) = res {
                        p.handle().close();
                    }
                    return;
                };
                match res {
                    Ok((peer, answer)) => {
                        // Avatar mode: video and audio in one MediaStream, so
                        // the browser plays them in sync (Reactor `ltx`: both
                        // tracks on one clock in a single MediaStream).
                        let answer = if live.table.mode == crate::engine::Mode::Avatar {
                            fastvideo_webrtc::sdp::unify_msid(&answer, "fv-avatar")
                        } else {
                            answer
                        };
                        c.answer = Answer::Ready(answer);
                        c.peer = Some(peer.handle().clone());
                        (peer, std::mem::take(&mut c.candidates), c.mapping.clone())
                    }
                    Err(e) => {
                        tracing::warn!(cid, error = %e, "reactor offer refused");
                        c.answer = Answer::Failed(e.to_string());
                        return;
                    }
                }
            };
            let handle = peer.handle().clone();
            gateway::spawn(me.clone(), live, cid, peer, seed, mapping);
            for cand in buffered {
                let _ = handle.add_remote_candidate(&cand).await;
            }
        });
        Ok(json!({"connection_id": cid}))
    }

    /// `GET sdp_params`.
    pub fn poll_answer(&self, sid: &str, cid: u32) -> Result<AnswerPoll, Refusal> {
        let live = self.require_running(sid)?;
        let mut conns = lock(&live.conns);
        let c = conns.get_mut(&cid).ok_or_else(|| Refusal::new(404, "Unknown connection"))?;
        match std::mem::replace(&mut c.answer, Answer::Taken) {
            Answer::Ready(a) => Ok(AnswerPoll::Ready(a)),
            Answer::Failed(e) => {
                c.answer = Answer::Failed(e.clone());
                Err(Refusal::new(400, format!("negotiation failed: {e}")))
            }
            other => {
                c.answer = other;
                Ok(AnswerPoll::Pending)
            }
        }
    }

    /// `POST ice_candidates`.
    pub async fn add_candidates(&self, sid: &str, cid: u32, body: Candidates) -> Result<(), Refusal> {
        let live = self.require_running(sid)?;
        let peer = {
            let mut conns = lock(&live.conns);
            let c = conns.get_mut(&cid).ok_or_else(|| Refusal::new(404, "Unknown connection"))?;
            match &c.peer {
                Some(p) => Some(p.clone()),
                None => {
                    for cand in body.candidates.iter().filter(|c| !c.candidate.trim().is_empty()) {
                        if c.candidates.len() < self.config().max_candidates {
                            c.candidates.push(cand.candidate.clone());
                        }
                    }
                    None
                }
            }
        };
        if let Some(p) = peer {
            for cand in body.candidates.iter().filter(|c| !c.candidate.trim().is_empty()) {
                let _ = p.add_remote_candidate(&cand.candidate).await;
            }
        }
        let _ = body.is_final;
        Ok(())
    }
}
