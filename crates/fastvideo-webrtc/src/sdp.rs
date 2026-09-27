//! SDP validation and munging (design §5.7, §5.8; WP-04).
//!
//! str0m does the real offer/answer work. This module is the thin, pure
//! text layer around it:
//!
//! - [`Sdp::parse`] and [`validate_offer`] reject malformed or unusable
//!   offers before they reach str0m, and summarise the m-lines (mid, kind,
//!   direction, codecs) so front-ends can map mids to track names.
//! - Munging helpers turn str0m's trickle-style output into the
//!   **non-trickle** SDP every front-end needs: all candidates embedded plus
//!   `a=end-of-candidates`, and no `a=ice-options:trickle`
//!   (reactor §3.3 "the server never trickles its own candidates"; fal §8.2
//!   "No trickle ICE").
//! - Offer-side helpers for the video-only case (audio m-line answered
//!   `a=inactive`, design §5.3), mDNS candidate stripping, Opus stereo, and
//!   H.264-first ordering for WHIP (streaming-refs §1.9).
//!
//! Everything here works on text and has no str0m dependency, so the tests
//! run in the default (CPU, no-feature) CI.

use std::fmt;
use std::net::IpAddr;

/// Largest SDP we accept from a client. Browser offers are a few KiB.
pub const MAX_SDP_BYTES: usize = 64 * 1024;

/// SDP errors. Messages are safe to show to clients.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SdpError {
    #[error("sdp is empty")]
    Empty,
    #[error("sdp is larger than {MAX_SDP_BYTES} bytes")]
    TooLarge,
    #[error("sdp must start with v=0")]
    MissingVersion,
    #[error("sdp has no m-lines")]
    NoMedia,
    #[error("malformed sdp line {line}: {text:?}")]
    Malformed { line: usize, text: String },
    #[error("m-line {index} has no a=mid")]
    MissingMid { index: usize },
    #[error("duplicate a=mid:{0}")]
    DuplicateMid(String),
    #[error("missing a=ice-ufrag / a=ice-pwd")]
    MissingIceCredentials,
    #[error("missing a=fingerprint")]
    MissingFingerprint,
    #[error("offer has no {0} codec this server can send ({1})")]
    NoCommonCodec(&'static str, &'static str),
}

/// Media direction attribute (`a=sendrecv` and friends).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    SendRecv,
    SendOnly,
    RecvOnly,
    Inactive,
}

impl Direction {
    pub fn attr(self) -> &'static str {
        match self {
            Direction::SendRecv => "sendrecv",
            Direction::SendOnly => "sendonly",
            Direction::RecvOnly => "recvonly",
            Direction::Inactive => "inactive",
        }
    }

    fn from_attr(s: &str) -> Option<Self> {
        Some(match s {
            "sendrecv" => Direction::SendRecv,
            "sendonly" => Direction::SendOnly,
            "recvonly" => Direction::RecvOnly,
            "inactive" => Direction::Inactive,
            _ => return None,
        })
    }

    /// The direction the other side of this m-line has.
    pub fn invert(self) -> Self {
        match self {
            Direction::SendOnly => Direction::RecvOnly,
            Direction::RecvOnly => Direction::SendOnly,
            d => d,
        }
    }

    /// True when the side that wrote this direction will receive media.
    pub fn is_receiving(self) -> bool {
        matches!(self, Direction::SendRecv | Direction::RecvOnly)
    }

    /// True when the side that wrote this direction will send media.
    pub fn is_sending(self) -> bool {
        matches!(self, Direction::SendRecv | Direction::SendOnly)
    }
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.attr())
    }
}

/// Kind of an m-line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Video,
    Audio,
    Application,
    Other,
}

/// One `a=rtpmap` (plus its `a=fmtp`) of an m-line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtpCodec {
    pub pt: u8,
    /// Encoding name as written (`H264`, `opus`, `rtx`, ...).
    pub name: String,
    pub clock_rate: u32,
    pub channels: Option<u8>,
    pub fmtp: Option<String>,
}

impl RtpCodec {
    pub fn is(&self, name: &str) -> bool {
        self.name.eq_ignore_ascii_case(name)
    }

    /// Value of one `key=value` fmtp parameter.
    pub fn fmtp_param(&self, key: &str) -> Option<&str> {
        self.fmtp.as_deref().and_then(|f| {
            f.split(';').find_map(|kv| {
                let (k, v) = kv.trim().split_once('=')?;
                k.trim().eq_ignore_ascii_case(key).then_some(v.trim())
            })
        })
    }

    /// H.264 we can send: FU-A (packetization-mode=1) and a Constrained
    /// Baseline or Baseline profile (`42xxxx`). Our encoder emits CB (§5.9).
    pub fn is_sendable_h264(&self) -> bool {
        if !self.is("H264") || self.clock_rate != 90_000 {
            return false;
        }
        let pm1 = self.fmtp_param("packetization-mode") == Some("1");
        let baseline = self
            .fmtp_param("profile-level-id")
            .is_some_and(|p| p.len() == 6 && p[..2].eq_ignore_ascii_case("42"));
        pm1 && baseline
    }

    pub fn is_opus(&self) -> bool {
        self.is("opus") && self.clock_rate == 48_000
    }
}

/// One m-section: the `m=` line and every line after it up to the next `m=`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSection {
    pub m_line: String,
    pub lines: Vec<String>,
}

impl MediaSection {
    fn m_fields(&self) -> Vec<&str> {
        self.m_line
            .strip_prefix("m=")
            .unwrap_or(&self.m_line)
            .split_whitespace()
            .collect()
    }

    pub fn kind(&self) -> MediaKind {
        match self.m_fields().first().copied() {
            Some("video") => MediaKind::Video,
            Some("audio") => MediaKind::Audio,
            Some("application") => MediaKind::Application,
            _ => MediaKind::Other,
        }
    }

    /// Port 0 means the m-line was rejected or stopped.
    pub fn is_rejected(&self) -> bool {
        self.m_fields().get(1).is_some_and(|p| *p == "0")
    }

    /// The format list of the m-line (payload types for RTP).
    pub fn formats(&self) -> Vec<String> {
        self.m_fields().iter().skip(3).map(|s| s.to_string()).collect()
    }

    /// Values of every `a=<name>:<value>` line (or `a=<name>` flags as "").
    pub fn attrs<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
        attr_values(&self.lines, name)
    }

    pub fn attr(&self, name: &str) -> Option<&str> {
        first_attr(&self.lines, name)
    }

    pub fn mid(&self) -> Option<&str> {
        self.attr("mid")
    }

    /// Direction of the m-line, defaulting to `sendrecv` (RFC 8866 §6.7).
    pub fn direction(&self) -> Direction {
        self.lines
            .iter()
            .filter_map(|l| l.strip_prefix("a="))
            .find_map(Direction::from_attr)
            .unwrap_or(Direction::SendRecv)
    }

    /// Replace (or add) the direction attribute.
    pub fn set_direction(&mut self, dir: Direction) {
        let new = format!("a={}", dir.attr());
        if let Some(l) = self.lines.iter_mut().find(|l| {
            l.strip_prefix("a=")
                .is_some_and(|a| Direction::from_attr(a).is_some())
        }) {
            *l = new;
        } else {
            // After a=mid if present, else at the end.
            let at = self
                .lines
                .iter()
                .position(|l| l.starts_with("a=mid:"))
                .map(|i| i + 1)
                .unwrap_or(self.lines.len());
            self.lines.insert(at, new);
        }
    }

    /// Codecs from `a=rtpmap` + `a=fmtp`, in m-line format order.
    pub fn codecs(&self) -> Vec<RtpCodec> {
        let mut out = Vec::new();
        for pt in self.formats() {
            let Ok(pt_num) = pt.parse::<u8>() else { continue };
            let prefix = format!("{pt} ");
            let Some(map) = self.attrs("rtpmap").find(|v| v.starts_with(&prefix)) else {
                continue;
            };
            let enc = &map[prefix.len()..];
            let mut parts = enc.split('/');
            let name = parts.next().unwrap_or_default().to_string();
            let clock_rate = parts.next().and_then(|c| c.parse().ok()).unwrap_or(0);
            let channels = parts.next().and_then(|c| c.parse().ok());
            let fmtp = self
                .attrs("fmtp")
                .find(|v| v.starts_with(&prefix))
                .map(|v| v[prefix.len()..].to_string());
            out.push(RtpCodec { pt: pt_num, name, clock_rate, channels, fmtp });
        }
        out
    }

    /// `a=candidate:` values (without the `a=` prefix, with `candidate:`).
    pub fn candidates(&self) -> impl Iterator<Item = &str> {
        self.lines
            .iter()
            .filter_map(|l| l.strip_prefix("a="))
            .filter(|l| l.starts_with("candidate:"))
    }
}

fn attr_values<'a>(lines: &'a [String], name: &'a str) -> impl Iterator<Item = &'a str> + 'a {
    lines.iter().filter_map(move |l| match_attr(l, name))
}

fn first_attr<'a>(lines: &'a [String], name: &str) -> Option<&'a str> {
    lines.iter().find_map(|l| match_attr(l, name))
}

fn match_attr<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let rest = line.strip_prefix("a=")?.strip_prefix(name)?;
    if rest.is_empty() {
        Some("")
    } else {
        rest.strip_prefix(':')
    }
}

/// A parsed SDP: session-level lines plus m-sections, preserving every line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sdp {
    pub session: Vec<String>,
    pub media: Vec<MediaSection>,
}

impl Sdp {
    /// Parse SDP text. Accepts CRLF or bare LF; tolerates trailing blanks.
    pub fn parse(text: &str) -> Result<Self, SdpError> {
        if text.trim().is_empty() {
            return Err(SdpError::Empty);
        }
        if text.len() > MAX_SDP_BYTES {
            return Err(SdpError::TooLarge);
        }
        let mut session = Vec::new();
        let mut media: Vec<MediaSection> = Vec::new();
        for (i, raw) in text.split('\n').enumerate() {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            let b = line.as_bytes();
            if b.len() < 2 || b[1] != b'=' || !b[0].is_ascii_lowercase() {
                return Err(SdpError::Malformed { line: i + 1, text: line.chars().take(80).collect() });
            }
            if line.starts_with("m=") {
                media.push(MediaSection { m_line: line.to_string(), lines: Vec::new() });
            } else if let Some(m) = media.last_mut() {
                m.lines.push(line.to_string());
            } else {
                session.push(line.to_string());
            }
        }
        if session.first().map(String::as_str) != Some("v=0") {
            return Err(SdpError::MissingVersion);
        }
        Ok(Sdp { session, media })
    }

    pub fn session_attr(&self, name: &str) -> Option<&str> {
        first_attr(&self.session, name)
    }

    /// A media attribute, falling back to the session level (ice-ufrag,
    /// ice-pwd, fingerprint and setup may be written at either level).
    pub fn effective_attr<'a>(&'a self, m: &'a MediaSection, name: &str) -> Option<&'a str> {
        m.attr(name).or_else(|| self.session_attr(name))
    }

    pub fn section_by_mid(&self, mid: &str) -> Option<&MediaSection> {
        self.media.iter().find(|m| m.mid() == Some(mid))
    }

    pub fn section_by_mid_mut(&mut self, mid: &str) -> Option<&mut MediaSection> {
        self.media.iter_mut().find(|m| m.mid() == Some(mid))
    }

    /// All candidate values in the SDP (session and media level).
    pub fn candidates(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self
            .session
            .iter()
            .filter_map(|l| l.strip_prefix("a="))
            .filter(|l| l.starts_with("candidate:"))
            .collect();
        for m in &self.media {
            v.extend(m.candidates());
        }
        v
    }

    /// True when the SDP carries `a=end-of-candidates` anywhere.
    pub fn has_end_of_candidates(&self) -> bool {
        attr_values(&self.session, "end-of-candidates").next().is_some()
            || self.media.iter().any(|m| m.attr("end-of-candidates").is_some())
    }

    /// Set the direction of every non-rejected m-line of `kind`.
    pub fn set_direction_for_kind(&mut self, kind: MediaKind, dir: Direction) -> usize {
        let mut n = 0;
        for m in self.media.iter_mut().filter(|m| m.kind() == kind && !m.is_rejected()) {
            m.set_direction(dir);
            n += 1;
        }
        n
    }

    /// Remove candidates we cannot use: mDNS (`*.local`) hostnames and
    /// anything whose address is not an IP literal. The peer is still found
    /// as peer-reflexive from its connectivity checks. Returns the count.
    pub fn strip_unresolvable_candidates(&mut self) -> usize {
        let keep = |l: &String| -> bool {
            let Some(c) = l.strip_prefix("a=candidate:") else { return true };
            c.split_whitespace().nth(4).is_some_and(|a| a.parse::<IpAddr>().is_ok())
        };
        let before = self.line_count();
        self.session.retain(keep);
        for m in &mut self.media {
            m.lines.retain(keep);
        }
        before - self.line_count()
    }

    fn line_count(&self) -> usize {
        self.session.len() + self.media.iter().map(|m| m.lines.len()).sum::<usize>()
    }

    /// Make the SDP non-trickle: drop `a=ice-options:trickle` and append
    /// `a=end-of-candidates` to every non-rejected m-section that carries
    /// candidates (or to the first one when candidates are only at session
    /// level or absent). Idempotent.
    pub fn finish_candidates(&mut self) {
        let not_trickle = |l: &String| {
            !(l.starts_with("a=ice-options:")
                && l["a=ice-options:".len()..].split_whitespace().all(|o| o == "trickle"))
        };
        let strip_trickle_opt = |lines: &mut Vec<String>| {
            lines.retain(not_trickle);
            for l in lines.iter_mut() {
                if let Some(opts) = l.strip_prefix("a=ice-options:") {
                    let rest: Vec<&str> =
                        opts.split_whitespace().filter(|o| *o != "trickle").collect();
                    *l = format!("a=ice-options:{}", rest.join(" "));
                }
            }
        };
        strip_trickle_opt(&mut self.session);
        let mut done_any = false;
        for m in self.media.iter_mut().filter(|m| !m.is_rejected()) {
            strip_trickle_opt(&mut m.lines);
            if m.candidates().next().is_none() {
                continue;
            }
            done_any = true;
            if m.attr("end-of-candidates").is_none() {
                let last = m.lines.iter().rposition(|l| l.starts_with("a=candidate:")).unwrap_or(0);
                m.lines.insert(last + 1, "a=end-of-candidates".to_string());
            }
        }
        if !done_any && !self.has_end_of_candidates() {
            if let Some(m) = self.media.iter_mut().find(|m| !m.is_rejected()) {
                m.lines.push("a=end-of-candidates".to_string());
            }
        }
    }

    /// Set `stereo=1;sprop-stereo=1` (or remove them) on every Opus fmtp.
    /// Chrome downmixes to mono unless the SDP it receives says stereo.
    pub fn set_opus_stereo(&mut self, stereo: bool) {
        for m in self.media.iter_mut().filter(|m| m.kind() == MediaKind::Audio) {
            let opus_pts: Vec<u8> = m.codecs().iter().filter(|c| c.is_opus()).map(|c| c.pt).collect();
            for pt in opus_pts {
                let prefix = format!("a=fmtp:{pt} ");
                let idx = m.lines.iter().position(|l| l.starts_with(&prefix));
                let current = idx.map(|i| m.lines[i][prefix.len()..].to_string()).unwrap_or_default();
                let mut params: Vec<String> = current
                    .split(';')
                    .map(|s| s.trim().to_string())
                    .filter(|s| {
                        !s.is_empty()
                            && !s.starts_with("stereo=")
                            && !s.starts_with("sprop-stereo=")
                    })
                    .collect();
                if stereo {
                    params.push("stereo=1".into());
                    params.push("sprop-stereo=1".into());
                }
                let line = format!("{prefix}{}", params.join(";"));
                match idx {
                    Some(i) if params.is_empty() => {
                        m.lines.remove(i);
                    }
                    Some(i) => m.lines[i] = line,
                    None if !params.is_empty() => {
                        let at = m
                            .lines
                            .iter()
                            .position(|l| l.starts_with(&format!("a=rtpmap:{pt} ")))
                            .map(|i| i + 1)
                            .unwrap_or(m.lines.len());
                        m.lines.insert(at, line);
                    }
                    None => {}
                }
            }
        }
    }

    /// Reorder every video m-line's format list so H.264 (and its RTX) come
    /// first. Cloudflare ingests H.264 only and accepts the first offered
    /// codec; VP8 there "negotiates fine but produces a black stream"
    /// (streaming-refs §1.9).
    pub fn prefer_h264(&mut self) {
        for m in self.media.iter_mut().filter(|m| m.kind() == MediaKind::Video) {
            let codecs = m.codecs();
            let h264: Vec<u8> = codecs.iter().filter(|c| c.is("H264")).map(|c| c.pt).collect();
            let rank = |pt: &str| -> u8 {
                let Ok(p) = pt.parse::<u8>() else { return 3 };
                if h264.contains(&p) {
                    return 0;
                }
                let is_h264_rtx = codecs.iter().any(|c| {
                    c.pt == p
                        && c.is("rtx")
                        && c.fmtp_param("apt").and_then(|a| a.parse::<u8>().ok()).is_some_and(|a| h264.contains(&a))
                });
                if is_h264_rtx { 1 } else { 2 }
            };
            let mut fields: Vec<String> = m.m_fields().iter().map(|s| s.to_string()).collect();
            if fields.len() <= 3 {
                continue;
            }
            let mut fmts = fields.split_off(3);
            fmts.sort_by_key(|f| rank(f)); // stable: keeps order within a rank
            fields.extend(fmts);
            m.m_line = format!("m={}", fields.join(" "));
        }
    }

    /// Name of the first non-RTX codec of the first video m-line (for logs).
    pub fn negotiated_video_codec(&self) -> Option<String> {
        let m = self.media.iter().find(|m| m.kind() == MediaKind::Video && !m.is_rejected())?;
        m.codecs().into_iter().find(|c| !c.is("rtx") && !c.is("red") && !c.is("ulpfec")).map(|c| c.name)
    }

    /// Remove a session-level attribute (e.g. `x-reactor-frame-metadata`).
    pub fn remove_session_attr(&mut self, name: &str) {
        let full = format!("a={name}");
        self.session.retain(|l| !(l == &full || l.starts_with(&format!("{full}:"))));
    }
}

impl fmt::Display for Sdp {
    /// Serialise with CRLF line endings (RFC 8866).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for l in &self.session {
            write!(f, "{l}\r\n")?;
        }
        for m in &self.media {
            write!(f, "{}\r\n", m.m_line)?;
            for l in &m.lines {
                write!(f, "{l}\r\n")?;
            }
        }
        Ok(())
    }
}

/// One m-line of a validated offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSummary {
    pub index: usize,
    pub mid: String,
    pub kind: MediaKind,
    /// Direction as written by the offerer.
    pub direction: Direction,
    pub rejected: bool,
    pub codecs: Vec<RtpCodec>,
}

impl MediaSummary {
    /// Whether the offerer wants to receive on this m-line.
    pub fn offerer_receives(&self) -> bool {
        !self.rejected && self.direction.is_receiving()
    }
}

/// What [`validate_offer`] learned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfferSummary {
    pub media: Vec<MediaSummary>,
    pub has_data_channel: bool,
    pub ice_lite: bool,
    /// Candidates embedded in the offer (a full, non-trickle offer has >0).
    pub candidate_count: usize,
    pub end_of_candidates: bool,
    /// Offer asked for the Reactor RXMT trailer (we never mirror it).
    pub reactor_frame_metadata: bool,
}

impl OfferSummary {
    pub fn first(&self, kind: MediaKind) -> Option<&MediaSummary> {
        self.media.iter().find(|m| m.kind == kind && !m.rejected)
    }
}

/// What the answering side needs from an offer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OfferRequirements {
    /// Some video m-line the offerer receives on must offer sendable H.264.
    pub video: bool,
    /// Some audio m-line the offerer receives on must offer Opus.
    pub audio: bool,
}

/// Validate an offer (client → us) and summarise it.
pub fn validate_offer(sdp: &Sdp, req: OfferRequirements) -> Result<OfferSummary, SdpError> {
    if sdp.media.is_empty() {
        return Err(SdpError::NoMedia);
    }
    let mut media = Vec::new();
    let mut mids = std::collections::HashSet::new();
    for (index, m) in sdp.media.iter().enumerate() {
        let rejected = m.is_rejected();
        let mid = match m.mid() {
            Some(mid) => mid.to_string(),
            None if rejected => String::new(),
            None => return Err(SdpError::MissingMid { index }),
        };
        if !mid.is_empty() && !mids.insert(mid.clone()) {
            return Err(SdpError::DuplicateMid(mid));
        }
        if !rejected {
            if sdp.effective_attr(m, "ice-ufrag").is_none() || sdp.effective_attr(m, "ice-pwd").is_none() {
                return Err(SdpError::MissingIceCredentials);
            }
            if sdp.effective_attr(m, "fingerprint").is_none() {
                return Err(SdpError::MissingFingerprint);
            }
        }
        media.push(MediaSummary {
            index,
            mid,
            kind: m.kind(),
            direction: m.direction(),
            rejected,
            codecs: m.codecs(),
        });
    }
    let recv = |k: MediaKind| media.iter().filter(move |m| m.kind == k && m.offerer_receives());
    if req.video && !recv(MediaKind::Video).any(|m| m.codecs.iter().any(RtpCodec::is_sendable_h264)) {
        return Err(SdpError::NoCommonCodec("video", "H.264 constrained baseline, packetization-mode=1"));
    }
    if req.audio && !recv(MediaKind::Audio).any(|m| m.codecs.iter().any(RtpCodec::is_opus)) {
        return Err(SdpError::NoCommonCodec("audio", "opus/48000"));
    }
    Ok(OfferSummary {
        has_data_channel: media.iter().any(|m| m.kind == MediaKind::Application && !m.rejected),
        ice_lite: sdp.session_attr("ice-lite").is_some(),
        candidate_count: sdp.candidates().len(),
        end_of_candidates: sdp.has_end_of_candidates(),
        reactor_frame_metadata: sdp.session_attr("x-reactor-frame-metadata").is_some(),
        media,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFER: &str = "v=0\r\n\
o=- 4611731400430051336 2 IN IP4 127.0.0.1\r\n\
s=-\r\n\
t=0 0\r\n\
a=group:BUNDLE 0 1 2\r\n\
a=extmap-allow-mixed\r\n\
a=msid-semantic: WMS\r\n\
m=video 9 UDP/TLS/RTP/SAVPF 96 97 102 103 108 109\r\n\
c=IN IP4 0.0.0.0\r\n\
a=rtcp:9 IN IP4 0.0.0.0\r\n\
a=candidate:1 1 udp 2122260223 192.168.1.5 50000 typ host generation 0\r\n\
a=candidate:2 1 udp 2122260223 3f1c6a2e-1111-2222-3333-444455556666.local 50001 typ host generation 0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:0123456789abcdefghijklmn\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 AA:BB\r\n\
a=setup:actpass\r\n\
a=mid:0\r\n\
a=recvonly\r\n\
a=rtcp-mux\r\n\
a=rtpmap:96 VP8/90000\r\n\
a=rtpmap:97 rtx/90000\r\n\
a=fmtp:97 apt=96\r\n\
a=rtpmap:102 H264/90000\r\n\
a=fmtp:102 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f\r\n\
a=rtpmap:103 rtx/90000\r\n\
a=fmtp:103 apt=102\r\n\
a=rtpmap:108 H264/90000\r\n\
a=fmtp:108 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f\r\n\
a=rtpmap:109 rtx/90000\r\n\
a=fmtp:109 apt=108\r\n\
m=audio 9 UDP/TLS/RTP/SAVPF 111 0\r\n\
c=IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:0123456789abcdefghijklmn\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 AA:BB\r\n\
a=setup:actpass\r\n\
a=mid:1\r\n\
a=recvonly\r\n\
a=rtpmap:111 opus/48000/2\r\n\
a=fmtp:111 minptime=10;useinbandfec=1\r\n\
a=rtpmap:0 PCMU/8000\r\n\
m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n\
c=IN IP4 0.0.0.0\r\n\
a=ice-ufrag:abcd\r\n\
a=ice-pwd:0123456789abcdefghijklmn\r\n\
a=ice-options:trickle\r\n\
a=fingerprint:sha-256 AA:BB\r\n\
a=setup:actpass\r\n\
a=mid:2\r\n\
a=sctp-port:5000\r\n\
a=max-message-size:262144\r\n";

    #[test]
    fn parses_and_round_trips() {
        let sdp = Sdp::parse(OFFER).unwrap();
        assert_eq!(sdp.media.len(), 3);
        assert_eq!(sdp.to_string(), OFFER);
        // LF-only input parses to the same thing.
        let lf = OFFER.replace("\r\n", "\n");
        assert_eq!(Sdp::parse(&lf).unwrap(), sdp);
    }

    #[test]
    fn summary_reports_mids_kinds_codecs() {
        let sdp = Sdp::parse(OFFER).unwrap();
        let s = validate_offer(&sdp, OfferRequirements { video: true, audio: true }).unwrap();
        assert_eq!(s.media.len(), 3);
        assert_eq!(s.media[0].mid, "0");
        assert_eq!(s.media[0].kind, MediaKind::Video);
        assert_eq!(s.media[0].direction, Direction::RecvOnly);
        assert_eq!(s.media[1].kind, MediaKind::Audio);
        assert!(s.has_data_channel);
        assert_eq!(s.candidate_count, 2);
        assert!(!s.end_of_candidates);
        let h264: Vec<_> = s.media[0].codecs.iter().filter(|c| c.is_sendable_h264()).map(|c| c.pt).collect();
        // 108 is packetization-mode=0: not sendable with FU-A.
        assert_eq!(h264, vec![102]);
        assert_eq!(s.media[1].codecs[0].channels, Some(2));
    }

    #[test]
    fn rejects_bad_offers() {
        assert_eq!(Sdp::parse(""), Err(SdpError::Empty));
        assert_eq!(Sdp::parse("o=- 1 1 IN IP4 0.0.0.0\r\n"), Err(SdpError::MissingVersion));
        assert!(matches!(Sdp::parse("v=0\r\nnot a line\r\n"), Err(SdpError::Malformed { line: 2, .. })));
        assert_eq!(Sdp::parse(&"v=0\r\n".repeat(20_000)), Err(SdpError::TooLarge));

        let no_media = Sdp::parse("v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\ns=-\r\nt=0 0\r\n").unwrap();
        assert_eq!(validate_offer(&no_media, OfferRequirements::default()), Err(SdpError::NoMedia));

        let no_fp = Sdp::parse(&OFFER.replace("a=fingerprint:sha-256 AA:BB\r\n", "")).unwrap();
        assert_eq!(validate_offer(&no_fp, OfferRequirements::default()), Err(SdpError::MissingFingerprint));

        let no_ice = Sdp::parse(&OFFER.replace("a=ice-pwd:0123456789abcdefghijklmn\r\n", "")).unwrap();
        assert_eq!(validate_offer(&no_ice, OfferRequirements::default()), Err(SdpError::MissingIceCredentials));

        let dup = Sdp::parse(&OFFER.replace("a=mid:1", "a=mid:0")).unwrap();
        assert_eq!(validate_offer(&dup, OfferRequirements::default()), Err(SdpError::DuplicateMid("0".into())));

        let vp8_only = Sdp::parse(&OFFER.replace("H264", "H263")).unwrap();
        assert!(matches!(
            validate_offer(&vp8_only, OfferRequirements { video: true, audio: false }),
            Err(SdpError::NoCommonCodec("video", _))
        ));
        let no_opus = Sdp::parse(&OFFER.replace("opus/48000/2", "G722/8000")).unwrap();
        assert!(matches!(
            validate_offer(&no_opus, OfferRequirements { video: false, audio: true }),
            Err(SdpError::NoCommonCodec("audio", _))
        ));
        // Video-only sessions don't care about the audio codec.
        assert!(validate_offer(&no_opus, OfferRequirements { video: true, audio: false }).is_ok());
    }

    #[test]
    fn session_level_credentials_are_enough() {
        let text = OFFER
            .replace("a=ice-ufrag:abcd\r\n", "")
            .replace("a=ice-pwd:0123456789abcdefghijklmn\r\n", "")
            .replace("a=fingerprint:sha-256 AA:BB\r\n", "")
            .replace(
                "t=0 0\r\n",
                "t=0 0\r\na=ice-ufrag:abcd\r\na=ice-pwd:0123456789abcdefghijklmn\r\na=fingerprint:sha-256 AA:BB\r\n",
            );
        let sdp = Sdp::parse(&text).unwrap();
        validate_offer(&sdp, OfferRequirements::default()).unwrap();
    }

    #[test]
    fn audio_inactive_for_video_only_sessions() {
        let mut sdp = Sdp::parse(OFFER).unwrap();
        assert_eq!(sdp.set_direction_for_kind(MediaKind::Audio, Direction::Inactive), 1);
        assert_eq!(sdp.media[1].direction(), Direction::Inactive);
        assert_eq!(sdp.media[0].direction(), Direction::RecvOnly);
        // Exactly one direction attribute remains.
        assert_eq!(sdp.media[1].lines.iter().filter(|l| *l == "a=inactive" || *l == "a=recvonly").count(), 1);
        // No direction attr present: it is inserted after a=mid.
        let mut m = MediaSection { m_line: "m=audio 9 X 0".into(), lines: vec!["a=mid:7".into(), "a=rtcp-mux".into()] };
        assert_eq!(m.direction(), Direction::SendRecv);
        m.set_direction(Direction::SendOnly);
        assert_eq!(m.lines, vec!["a=mid:7", "a=sendonly", "a=rtcp-mux"]);
    }

    #[test]
    fn strips_mdns_candidates() {
        let mut sdp = Sdp::parse(OFFER).unwrap();
        assert_eq!(sdp.strip_unresolvable_candidates(), 1);
        assert_eq!(sdp.candidates(), vec!["candidate:1 1 udp 2122260223 192.168.1.5 50000 typ host generation 0"]);
    }

    #[test]
    fn finish_candidates_makes_sdp_non_trickle() {
        let mut sdp = Sdp::parse(OFFER).unwrap();
        sdp.finish_candidates();
        let text = sdp.to_string();
        assert!(!text.contains("a=ice-options:trickle"));
        assert!(sdp.has_end_of_candidates());
        // end-of-candidates follows the last candidate of the first section.
        let m0 = &sdp.media[0].lines;
        let last_c = m0.iter().rposition(|l| l.starts_with("a=candidate:")).unwrap();
        assert_eq!(m0[last_c + 1], "a=end-of-candidates");
        assert_eq!(text.matches("a=end-of-candidates").count(), 1);
        // Idempotent.
        let again = {
            let mut s = sdp.clone();
            s.finish_candidates();
            s
        };
        assert_eq!(again, sdp);

        // No candidates at all: still terminated (an empty, complete gather).
        let mut none = Sdp::parse(&OFFER.lines().filter(|l| !l.starts_with("a=candidate")).collect::<Vec<_>>().join("\r\n")).unwrap();
        none.finish_candidates();
        assert_eq!(none.media[0].lines.last().unwrap(), "a=end-of-candidates");

        // Other ice-options survive.
        let mut renomination = Sdp::parse(&OFFER.replacen("a=ice-options:trickle", "a=ice-options:trickle renomination", 1)).unwrap();
        renomination.finish_candidates();
        assert!(renomination.to_string().contains("a=ice-options:renomination\r\n"));
    }

    #[test]
    fn opus_stereo_munging() {
        let mut sdp = Sdp::parse(OFFER).unwrap();
        sdp.set_opus_stereo(true);
        assert!(sdp.media[1].lines.contains(&"a=fmtp:111 minptime=10;useinbandfec=1;stereo=1;sprop-stereo=1".to_string()));
        sdp.set_opus_stereo(true);
        assert_eq!(sdp.to_string().matches("stereo=1;sprop-stereo=1").count(), 1);
        sdp.set_opus_stereo(false);
        assert!(sdp.media[1].lines.contains(&"a=fmtp:111 minptime=10;useinbandfec=1".to_string()));
        // No fmtp line yet: one is added right after the rtpmap.
        let mut bare = Sdp::parse(&OFFER.replace("a=fmtp:111 minptime=10;useinbandfec=1\r\n", "")).unwrap();
        bare.set_opus_stereo(true);
        let l = &bare.media[1].lines;
        let i = l.iter().position(|x| x == "a=rtpmap:111 opus/48000/2").unwrap();
        assert_eq!(l[i + 1], "a=fmtp:111 stereo=1;sprop-stereo=1");
    }

    #[test]
    fn h264_first_ordering() {
        let mut sdp = Sdp::parse(OFFER).unwrap();
        assert_eq!(sdp.negotiated_video_codec().as_deref(), Some("VP8"));
        sdp.prefer_h264();
        assert_eq!(sdp.media[0].m_line, "m=video 9 UDP/TLS/RTP/SAVPF 102 108 103 109 96 97");
        assert_eq!(sdp.negotiated_video_codec().as_deref(), Some("H264"));
        // Audio untouched.
        assert_eq!(sdp.media[1].m_line, "m=audio 9 UDP/TLS/RTP/SAVPF 111 0");
    }

    #[test]
    fn reactor_metadata_attr_is_detected_and_removable() {
        let text = OFFER.replace("a=extmap-allow-mixed\r\n", "a=extmap-allow-mixed\r\na=x-reactor-frame-metadata:1\r\n");
        let mut sdp = Sdp::parse(&text).unwrap();
        assert!(validate_offer(&sdp, OfferRequirements::default()).unwrap().reactor_frame_metadata);
        sdp.remove_session_attr("x-reactor-frame-metadata");
        assert!(!validate_offer(&sdp, OfferRequirements::default()).unwrap().reactor_frame_metadata);
    }

    #[test]
    fn rejected_mlines_are_skipped() {
        let text = OFFER.replace("m=audio 9 ", "m=audio 0 ");
        let sdp = Sdp::parse(&text).unwrap();
        let s = validate_offer(&sdp, OfferRequirements { video: true, audio: false }).unwrap();
        assert!(s.media[1].rejected);
        assert!(s.first(MediaKind::Audio).is_none());
        assert!(matches!(
            validate_offer(&sdp, OfferRequirements { video: true, audio: true }),
            Err(SdpError::NoCommonCodec("audio", _))
        ));
    }
}
