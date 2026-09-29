//! A small ISO-BMFF (MP4) box reader: enough to prove the facts clients
//! depend on without ffprobe — `moov` before `mdat` (faststart), the track
//! list, codecs, H.264 profile/level (`avcC`), AAC object type, rate and
//! channels (`esds`), timescales and frame rate (`stts`).

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{MediaError, Result};

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AvcInfo {
    pub profile_idc: u8,
    pub constraint_flags: u8,
    pub level_idc: u8,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: u16,
    /// MPEG-4 audio object type from the AudioSpecificConfig (2 = AAC-LC).
    pub object_type: Option<u8>,
    /// ESDS objectTypeIndication (0x40 = MPEG-4 audio).
    pub object_type_indication: Option<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum TrackKind {
    Video,
    Audio,
    Other,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct TrackInfo {
    pub kind: TrackKind,
    pub handler: String,
    /// Sample entry fourcc (`avc1`, `mp4a`, `Opus`, ...).
    pub codec: String,
    pub timescale: u32,
    pub duration: u64,
    pub samples: u64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Constant frame rate when `stts` has one delta.
    pub fps: Option<f64>,
    pub avc: Option<AvcInfo>,
    pub audio: Option<AudioInfo>,
}

impl TrackInfo {
    pub fn duration_s(&self) -> f64 {
        if self.timescale == 0 { 0.0 } else { self.duration as f64 / f64::from(self.timescale) }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct Mp4Info {
    pub major_brand: String,
    /// Top-level box types in file order.
    pub top_level: Vec<String>,
    /// `moov` comes before the first `mdat`.
    pub faststart: bool,
    pub duration_s: f64,
    pub tracks: Vec<TrackInfo>,
}

impl Mp4Info {
    pub fn video(&self) -> Option<&TrackInfo> {
        self.tracks.iter().find(|t| t.kind == TrackKind::Video)
    }
    pub fn audio(&self) -> Option<&TrackInfo> {
        self.tracks.iter().find(|t| t.kind == TrackKind::Audio)
    }

    /// What fal clients see from hosted H3 (fal §6): faststart MP4, H.264
    /// video at 24 fps, and AAC-LC stereo 32 kHz audio. Returns the list of
    /// deviations (empty when conforming).
    pub fn fal_h3_problems(&self) -> Vec<String> {
        let mut p = Vec::new();
        if !self.faststart {
            p.push("moov is not before mdat (not faststart)".into());
        }
        match self.video() {
            None => p.push("no video track".into()),
            Some(v) => {
                if v.codec != "avc1" && v.codec != "avc3" {
                    p.push(format!("video codec {} is not H.264", v.codec));
                }
                match v.fps {
                    Some(f) if (f - 24.0).abs() < 1e-3 => {}
                    other => p.push(format!("video fps {other:?} is not 24")),
                }
            }
        }
        match self.audio() {
            None => p.push("no audio track".into()),
            Some(a) => {
                if a.codec != "mp4a" {
                    p.push(format!("audio codec {} is not AAC", a.codec));
                }
                match &a.audio {
                    None => p.push("audio sample entry unreadable".into()),
                    Some(ai) => {
                        if ai.object_type != Some(2) {
                            p.push(format!("AAC object type {:?} is not LC (2)", ai.object_type));
                        }
                        if ai.sample_rate != 32_000 {
                            p.push(format!("audio rate {} is not 32000", ai.sample_rate));
                        }
                        if ai.channels != 2 {
                            p.push(format!("audio channels {} is not 2", ai.channels));
                        }
                    }
                }
            }
        }
        p
    }
}

fn be16(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|s| u16::from_be_bytes([s[0], s[1]]))
}
fn be32(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}
fn be64(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o + 8).map(|s| u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
}
fn fourcc(b: &[u8]) -> String {
    b.iter().map(|&c| if c.is_ascii_graphic() || c == b' ' { c as char } else { '?' }).collect()
}

/// Child boxes of a container payload: (type, payload).
fn children(b: &[u8]) -> Vec<(String, &[u8])> {
    let mut out = Vec::new();
    let mut o = 0usize;
    while o + 8 <= b.len() {
        let size32 = be32(b, o).unwrap_or(0) as u64;
        let ty = fourcc(&b[o + 4..o + 8]);
        let (hdr, size) = match size32 {
            1 => match be64(b, o + 8) {
                Some(s) => (16u64, s),
                None => break,
            },
            0 => (8, (b.len() - o) as u64),
            s => (8, s),
        };
        if size < hdr || o as u64 + size > b.len() as u64 {
            break;
        }
        out.push((ty, &b[o + hdr as usize..o + size as usize]));
        o += size as usize;
    }
    out
}

fn child<'a>(b: &'a [u8], ty: &str) -> Option<&'a [u8]> {
    children(b).into_iter().find(|(t, _)| t == ty).map(|(_, p)| p)
}

fn descriptor(b: &[u8], o: &mut usize) -> Option<(u8, usize)> {
    let tag = *b.get(*o)?;
    *o += 1;
    let mut len = 0usize;
    for _ in 0..4 {
        let c = *b.get(*o)?;
        *o += 1;
        len = (len << 7) | usize::from(c & 0x7f);
        if c & 0x80 == 0 {
            break;
        }
    }
    Some((tag, len))
}

/// (objectTypeIndication, AudioSpecificConfig object type) from an `esds` payload.
fn parse_esds(b: &[u8]) -> (Option<u8>, Option<u8>) {
    let mut o = 4; // full box header
    let Some((3, _)) = descriptor(b, &mut o) else { return (None, None) };
    o += 2; // ES_ID
    let Some(&flags) = b.get(o) else { return (None, None) };
    o += 1;
    if flags & 0x80 != 0 {
        o += 2;
    }
    if flags & 0x40 != 0 {
        let n = usize::from(*b.get(o).unwrap_or(&0));
        o += 1 + n;
    }
    if flags & 0x20 != 0 {
        o += 2;
    }
    let Some((4, _)) = descriptor(b, &mut o) else { return (None, None) };
    let oti = b.get(o).copied();
    o += 13;
    let aot = match descriptor(b, &mut o) {
        Some((5, _)) => b.get(o).map(|c| c >> 3),
        _ => None,
    };
    (oti, aot)
}

fn parse_trak(t: &[u8]) -> Option<TrackInfo> {
    let mdia = child(t, "mdia")?;
    let mdhd = child(mdia, "mdhd")?;
    let (timescale, duration) = if mdhd.first() == Some(&1) {
        (be32(mdhd, 20)?, be64(mdhd, 24)?)
    } else {
        (be32(mdhd, 12)?, u64::from(be32(mdhd, 16)?))
    };
    let handler = child(mdia, "hdlr").map(|h| fourcc(h.get(8..12).unwrap_or(b"????"))).unwrap_or_default();
    let kind = match handler.as_str() {
        "vide" => TrackKind::Video,
        "soun" => TrackKind::Audio,
        _ => TrackKind::Other,
    };
    let stbl = child(child(mdia, "minf")?, "stbl")?;
    let mut info = TrackInfo {
        kind,
        handler,
        codec: String::new(),
        timescale,
        duration,
        samples: 0,
        width: None,
        height: None,
        fps: None,
        avc: None,
        audio: None,
    };
    if let Some(stts) = child(stbl, "stts") {
        let n = be32(stts, 4).unwrap_or(0) as usize;
        let mut deltas = Vec::new();
        for i in 0..n {
            let c = be32(stts, 8 + i * 8).unwrap_or(0);
            let d = be32(stts, 12 + i * 8).unwrap_or(0);
            info.samples += u64::from(c);
            deltas.push(d);
        }
        if kind == TrackKind::Video {
            // A trailing single-sample entry may carry a different delta.
            let main = deltas.first().copied().unwrap_or(0);
            if main > 0 && deltas.iter().take(deltas.len().saturating_sub(1).max(1)).all(|&d| d == main) {
                info.fps = Some(f64::from(timescale) / f64::from(main));
            }
        }
    }
    if let Some(stsd) = child(stbl, "stsd") {
        if let Some((ty, entry)) = children(stsd.get(8..)?).into_iter().next() {
            info.codec = ty.clone();
            match kind {
                TrackKind::Video if entry.len() >= 78 => {
                    info.width = be16(entry, 24).map(u32::from);
                    info.height = be16(entry, 26).map(u32::from);
                    if let Some(avcc) = child(&entry[78..], "avcC") {
                        if avcc.len() >= 4 {
                            info.avc = Some(AvcInfo { profile_idc: avcc[1], constraint_flags: avcc[2], level_idc: avcc[3] });
                        }
                    }
                }
                TrackKind::Audio if entry.len() >= 28 => {
                    let channels = be16(entry, 16).unwrap_or(0);
                    let sample_rate = be32(entry, 24).unwrap_or(0) >> 16;
                    let (oti, aot) = child(&entry[28..], "esds").map(parse_esds).unwrap_or((None, None));
                    info.audio =
                        Some(AudioInfo { sample_rate, channels, object_type: aot, object_type_indication: oti });
                }
                _ => {}
            }
        }
    }
    Some(info)
}

/// Inspect an MP4 file on disk. Only `moov` is read into memory.
pub fn inspect(path: &Path) -> Result<Mp4Info> {
    let perr = |m: &str| MediaError::parse(Some(path), m.to_string());
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    let mut top = Vec::new();
    let mut moov: Option<Vec<u8>> = None;
    let mut brand = String::new();
    let mut seen_mdat = false;
    let mut faststart = false;
    let mut pos = 0u64;
    while pos + 8 <= len {
        f.seek(SeekFrom::Start(pos))?;
        let mut h = [0u8; 16];
        let n = f.read(&mut h)?;
        if n < 8 {
            break;
        }
        let size32 = u64::from(u32::from_be_bytes([h[0], h[1], h[2], h[3]]));
        let ty = fourcc(&h[4..8]);
        let (hdr, size) = match size32 {
            1 if n >= 16 => (16, u64::from_be_bytes([h[8], h[9], h[10], h[11], h[12], h[13], h[14], h[15]])),
            0 => (8, len - pos),
            s => (8, s),
        };
        if size < hdr || pos + size > len {
            return Err(perr(&format!("box {ty} at {pos} overruns the file")));
        }
        match ty.as_str() {
            "moov" => {
                if !seen_mdat {
                    faststart = true;
                }
                if size > 64 << 20 {
                    return Err(perr("moov larger than 64 MiB"));
                }
                let mut b = vec![0u8; (size - hdr) as usize];
                f.seek(SeekFrom::Start(pos + hdr))?;
                f.read_exact(&mut b)?;
                moov = Some(b);
            }
            "mdat" => seen_mdat = true,
            "ftyp" => brand = fourcc(&h[8..12]),
            _ => {}
        }
        top.push(ty);
        pos += size;
    }
    let moov = moov.ok_or_else(|| perr("no moov box"))?;
    let duration_s = child(&moov, "mvhd")
        .and_then(|m| {
            if m.first() == Some(&1) {
                Some((be32(m, 20)?, be64(m, 24)?))
            } else {
                Some((be32(m, 12)?, u64::from(be32(m, 16)?)))
            }
        })
        .map(|(ts, d)| if ts == 0 { 0.0 } else { d as f64 / f64::from(ts) })
        .unwrap_or(0.0);
    let tracks = children(&moov).into_iter().filter(|(t, _)| t == "trak").filter_map(|(_, b)| parse_trak(b)).collect();
    Ok(Mp4Info { major_brand: brand, top_level: top, faststart, duration_s, tracks })
}

/// A header-only MP4 (`ftyp`, `moov`, an empty `mdat`) with one H.264 video
/// track of `frames` samples at `timescale / delta` fps and, optionally, an
/// AAC-LC stereo track at `audio_rate`. [`inspect`] reads it like a real
/// file; nothing can decode it. For tests of ingestion and negotiation that
/// must not need ffmpeg.
#[doc(hidden)]
pub fn header_only_mp4(width: u16, height: u16, timescale: u32, delta: u32, frames: u32, audio_rate: Option<u32>) -> Vec<u8> {
    fn bx(ty: &str, payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(ty.as_bytes());
        v.extend_from_slice(payload);
        v
    }
    fn full(rest: &[u8]) -> Vec<u8> {
        let mut v = 0u32.to_be_bytes().to_vec();
        v.extend_from_slice(rest);
        v
    }
    fn trak(handler: &str, timescale: u32, duration: u32, entry: Vec<u8>, stts: &[(u32, u32)]) -> Vec<u8> {
        let mut mdhd = vec![0u8; 8];
        mdhd.extend(timescale.to_be_bytes());
        mdhd.extend(duration.to_be_bytes());
        mdhd.extend([0u8; 4]);
        let mut hdlr = vec![0u8; 4];
        hdlr.extend(handler.as_bytes());
        hdlr.extend([0u8; 13]);
        let mut st = (stts.len() as u32).to_be_bytes().to_vec();
        for (c, d) in stts {
            st.extend(c.to_be_bytes());
            st.extend(d.to_be_bytes());
        }
        let stsd = full(&[&1u32.to_be_bytes()[..], &entry].concat());
        let stbl = bx("stbl", &[bx("stsd", &stsd), bx("stts", &full(&st))].concat());
        let minf = bx("minf", &stbl);
        let mdia = bx("mdia", &[bx("mdhd", &full(&mdhd)), bx("hdlr", &full(&hdlr)), minf].concat());
        bx("trak", &mdia)
    }
    let mut v = vec![0u8; 78];
    v[24..26].copy_from_slice(&width.to_be_bytes());
    v[26..28].copy_from_slice(&height.to_be_bytes());
    v.extend(bx("avcC", &[1, 100, 0, 40, 0xff]));
    let vdur = frames.saturating_mul(delta);
    let mut traks = trak("vide", timescale, vdur, bx("avc1", &v), &[(frames, delta)]);
    let secs = f64::from(vdur) / f64::from(timescale.max(1));
    if let Some(rate) = audio_rate {
        let mut a = vec![0u8; 28];
        a[16..18].copy_from_slice(&2u16.to_be_bytes());
        a[24..28].copy_from_slice(&(rate << 16).to_be_bytes());
        let esds = full(&[3, 25, 0, 1, 0, 4, 17, 0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0x12, 0x10, 6, 1, 2]);
        a.extend(bx("esds", &esds));
        let packets = (secs * f64::from(rate) / 1024.0).ceil() as u32;
        traks.extend(trak("soun", rate, packets * 1024, bx("mp4a", &a), &[(packets, 1024)]));
    }
    let mut mvhd = vec![0u8; 8];
    mvhd.extend(1000u32.to_be_bytes());
    mvhd.extend(((secs * 1000.0).round() as u32).to_be_bytes());
    let moov = bx("moov", &[bx("mvhd", &full(&mvhd)), traks].concat());
    let ftyp = bx("ftyp", b"isom\0\0\x02\0isomiso2avc1mp41");
    [ftyp, moov, bx("mdat", &[])].concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(ty: &str, payload: &[u8]) -> Vec<u8> {
        let mut v = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        v.extend_from_slice(ty.as_bytes());
        v.extend_from_slice(payload);
        v
    }

    fn full(ver_flags: u32, rest: &[u8]) -> Vec<u8> {
        let mut v = ver_flags.to_be_bytes().to_vec();
        v.extend_from_slice(rest);
        v
    }

    fn trak(handler: &str, timescale: u32, entry: Vec<u8>, stts: &[(u32, u32)]) -> Vec<u8> {
        let mut mdhd = vec![0u8; 12];
        mdhd.extend(timescale.to_be_bytes());
        mdhd.extend(1000u32.to_be_bytes());
        mdhd.extend([0u8; 4]);
        let mut hdlr = vec![0u8; 8];
        hdlr.extend(handler.as_bytes());
        hdlr.extend([0u8; 13]);
        let mut st = (stts.len() as u32).to_be_bytes().to_vec();
        for (c, d) in stts {
            st.extend(c.to_be_bytes());
            st.extend(d.to_be_bytes());
        }
        let stsd = full(0, &[&1u32.to_be_bytes()[..], &entry].concat());
        let stbl = bx("stbl", &[bx("stsd", &stsd), bx("stts", &full(0, &st))].concat());
        let minf = bx("minf", &stbl);
        let mdia = bx("mdia", &[bx("mdhd", &full(0, &mdhd[4..])), bx("hdlr", &full(0, &hdlr[4..])), minf].concat());
        bx("trak", &mdia)
    }

    fn synthetic(faststart: bool) -> Vec<u8> {
        // avc1 entry: 78 bytes of VisualSampleEntry then avcC (CB 3.2).
        let mut v = vec![0u8; 78];
        v[24..26].copy_from_slice(&1344u16.to_be_bytes());
        v[26..28].copy_from_slice(&768u16.to_be_bytes());
        v.extend(bx("avcC", &[1, 66, 0xc0, 32, 0xff]));
        let avc1 = bx("avc1", &v);
        // mp4a entry: 28 bytes of AudioSampleEntry then esds (AAC-LC, 32 kHz stereo).
        let mut a = vec![0u8; 28];
        a[16..18].copy_from_slice(&2u16.to_be_bytes());
        a[24..28].copy_from_slice(&(32_000u32 << 16).to_be_bytes());
        let esds = full(0, &[3, 25, 0, 1, 0, 4, 17, 0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0x12, 0x90, 6, 1, 2]);
        a.extend(bx("esds", &esds));
        let mp4a = bx("mp4a", &a);
        let mut mvhd = vec![0u8; 12];
        mvhd.extend(1000u32.to_be_bytes());
        mvhd.extend(5167u32.to_be_bytes());
        let moov = bx(
            "moov",
            &[
                bx("mvhd", &mvhd),
                trak("vide", 24_000, avc1, &[(124, 1000)]),
                trak("soun", 32_000, mp4a, &[(163, 1024)]),
            ]
            .concat(),
        );
        let ftyp = bx("ftyp", b"isom\0\0\x02\0isomiso2avc1mp41");
        let mdat = bx("mdat", &[0u8; 64]);
        if faststart { [ftyp, moov, mdat].concat() } else { [ftyp, mdat, moov].concat() }
    }

    #[test]
    fn reads_a_fal_shaped_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a.mp4");
        std::fs::write(&p, synthetic(true)).unwrap();
        let i = inspect(&p).unwrap();
        assert!(i.faststart);
        assert_eq!(i.top_level, vec!["ftyp", "moov", "mdat"]);
        assert_eq!(i.major_brand, "isom");
        assert!((i.duration_s - 5.167).abs() < 1e-9);
        let v = i.video().unwrap();
        assert_eq!((v.codec.as_str(), v.width, v.height, v.samples), ("avc1", Some(1344), Some(768), 124));
        assert_eq!(v.fps, Some(24.0));
        assert_eq!(v.avc, Some(AvcInfo { profile_idc: 66, constraint_flags: 0xc0, level_idc: 32 }));
        let a = i.audio().unwrap();
        assert_eq!(a.codec, "mp4a");
        let ai = a.audio.as_ref().unwrap();
        assert_eq!((ai.sample_rate, ai.channels, ai.object_type, ai.object_type_indication), (32_000, 2, Some(2), Some(0x40)));
        assert!(i.fal_h3_problems().is_empty(), "{:?}", i.fal_h3_problems());
    }

    #[test]
    fn detects_non_faststart() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("b.mp4");
        std::fs::write(&p, synthetic(false)).unwrap();
        let i = inspect(&p).unwrap();
        assert!(!i.faststart);
        assert_eq!(i.fal_h3_problems().len(), 1);
    }

    #[test]
    fn rejects_garbage() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("c.mp4");
        std::fs::write(&p, b"\0\0\0\x10junkjunkjunk").unwrap();
        assert!(inspect(&p).is_err());
    }

    #[test]
    fn header_only_files_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("h.mp4");
        std::fs::write(&p, header_only_mp4(768, 512, 24_000, 1000, 121, Some(44_100))).unwrap();
        let i = inspect(&p).unwrap();
        let v = i.video().unwrap();
        assert_eq!((v.width, v.height, v.samples, v.fps), (Some(768), Some(512), 121, Some(24.0)));
        assert!((v.duration_s() - 121.0 / 24.0).abs() < 1e-9);
        assert_eq!(i.audio().unwrap().audio.as_ref().unwrap().sample_rate, 44_100);
        std::fs::write(&p, header_only_mp4(640, 360, 30_000, 1001, 90, None)).unwrap();
        let i = inspect(&p).unwrap();
        assert!((i.video().unwrap().fps.unwrap() - 29.97).abs() < 1e-2);
        assert!(i.audio().is_none());
    }
}
