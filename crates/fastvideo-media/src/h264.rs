//! H.264 bitstream helpers: Annex-B splitting, SPS facts, and the level table.
//!
//! Level choice matters here (design §5.9, risk R3): `42e01f` (Constrained
//! Baseline, level 3.1) caps a frame at 3600 macroblocks, and 1344×768 is
//! 4032. The smallest level that carries 1344×768 at 24 fps is **3.2**
//! (5120 MBs, 216000 MB/s); 4.0 also fits. [`H264Level::min_for`] computes it.

use crate::error::{MediaError, Result};

/// NAL unit types we care about.
pub mod nal {
    pub const SLICE: u8 = 1;
    pub const IDR: u8 = 5;
    pub const SEI: u8 = 6;
    pub const SPS: u8 = 7;
    pub const PPS: u8 = 8;
    pub const AUD: u8 = 9;
}

/// Split an Annex-B byte stream into NAL units (start codes removed).
pub fn split_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = if k + 1 < starts.len() { starts[k + 1] - 3 } else { data.len() };
        // Trailing zero bytes belong to the next 4-byte start code.
        while e > s && data[e - 1] == 0 {
            e -= 1;
        }
        if e > s {
            out.push(&data[s..e]);
        }
    }
    out
}

pub fn nal_type(nal: &[u8]) -> u8 {
    nal.first().map(|b| b & 0x1f).unwrap_or(0)
}

/// Whether an access unit contains an IDR slice.
pub fn is_idr(annexb: &[u8]) -> bool {
    split_annexb(annexb).iter().any(|n| nal_type(n) == nal::IDR)
}

/// Annex-B to AVCC (4-byte big-endian length prefixes), dropping AUDs.
pub fn annexb_to_avcc(annexb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(annexb.len() + 16);
    for n in split_annexb(annexb) {
        if nal_type(n) == nal::AUD {
            continue;
        }
        out.extend_from_slice(&(n.len() as u32).to_be_bytes());
        out.extend_from_slice(n);
    }
    out
}

/// H.264 levels with their MaxMBPS / MaxFS limits (Table A-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum H264Level {
    L3_0,
    L3_1,
    L3_2,
    L4_0,
    L4_1,
    L4_2,
    L5_0,
    L5_1,
    L5_2,
}

impl H264Level {
    pub const ALL: [H264Level; 9] = [
        H264Level::L3_0,
        H264Level::L3_1,
        H264Level::L3_2,
        H264Level::L4_0,
        H264Level::L4_1,
        H264Level::L4_2,
        H264Level::L5_0,
        H264Level::L5_1,
        H264Level::L5_2,
    ];

    /// `level_idc` as written in the SPS (and the last byte of `profile-level-id`).
    pub fn idc(self) -> u8 {
        match self {
            H264Level::L3_0 => 30,
            H264Level::L3_1 => 31,
            H264Level::L3_2 => 32,
            H264Level::L4_0 => 40,
            H264Level::L4_1 => 41,
            H264Level::L4_2 => 42,
            H264Level::L5_0 => 50,
            H264Level::L5_1 => 51,
            H264Level::L5_2 => 52,
        }
    }

    pub fn from_idc(idc: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.idc() == idc)
    }

    /// (MaxMBPS, MaxFS) in macroblocks.
    pub fn limits(self) -> (u64, u64) {
        match self {
            H264Level::L3_0 => (40_500, 1_620),
            H264Level::L3_1 => (108_000, 3_600),
            H264Level::L3_2 => (216_000, 5_120),
            H264Level::L4_0 | H264Level::L4_1 => (245_760, 8_192),
            H264Level::L4_2 => (522_240, 8_704),
            H264Level::L5_0 => (589_824, 22_080),
            H264Level::L5_1 => (983_040, 36_864),
            H264Level::L5_2 => (2_073_600, 36_864),
        }
    }

    /// Whether `w×h` at `fps` fits this level (frame size, rate, and the
    /// per-dimension bound `sqrt(8·MaxFS)` macroblocks).
    pub fn fits(self, width: u32, height: u32, fps: u32) -> bool {
        let (mbps, fs) = self.limits();
        let mw = u64::from(width.div_ceil(16));
        let mh = u64::from(height.div_ceil(16));
        let frame = mw * mh;
        let dim_max = ((8 * fs) as f64).sqrt() as u64;
        frame <= fs && frame * u64::from(fps) <= mbps && mw <= dim_max && mh <= dim_max
    }

    /// The smallest level that carries `w×h` at `fps`.
    pub fn min_for(width: u32, height: u32, fps: u32) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.fits(width, height, fps))
    }

    /// SDP `profile-level-id` for Constrained Baseline at this level
    /// (`42e0` + level), e.g. `42e01f` for 3.1 and `42e020` for 3.2.
    pub fn cb_profile_level_id(self) -> String {
        format!("42e0{:02x}", self.idc())
    }
}

impl std::fmt::Display for H264Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let idc = self.idc();
        write!(f, "{}.{}", idc / 10, idc % 10)
    }
}

impl std::str::FromStr for H264Level {
    type Err = MediaError;
    fn from_str(s: &str) -> Result<Self> {
        let t = s.trim().trim_start_matches(['L', 'l']).replace(['_', '.'], "");
        let idc: u8 = t.parse().map_err(|_| MediaError::invalid(format!("unknown H.264 level {s:?}")))?;
        let idc = if idc < 10 { idc * 10 } else { idc };
        Self::from_idc(idc).ok_or_else(|| MediaError::invalid(format!("unsupported H.264 level {s:?}")))
    }
}

/// Facts read from a sequence parameter set.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SpsInfo {
    pub profile_idc: u8,
    /// constraint_set0..5 flags, MSB first (as in `profile-level-id`'s middle byte).
    pub constraint_flags: u8,
    pub level_idc: u8,
    pub width: u32,
    pub height: u32,
}

impl SpsInfo {
    /// Constrained Baseline: profile 66 with constraint_set1 (A.2.1.1).
    pub fn is_constrained_baseline(&self) -> bool {
        self.profile_idc == 66 && self.constraint_flags & 0x40 != 0
    }
    pub fn profile_level_id(&self) -> String {
        format!("{:02x}{:02x}{:02x}", self.profile_idc, self.constraint_flags, self.level_idc)
    }
}

struct Bits {
    data: Vec<u8>,
    pos: usize,
}

impl Bits {
    fn new(nal_payload: &[u8]) -> Self {
        // Remove emulation-prevention bytes (00 00 03 -> 00 00).
        let mut data = Vec::with_capacity(nal_payload.len());
        let mut zeros = 0;
        for &b in nal_payload {
            if zeros >= 2 && b == 3 {
                zeros = 0;
                continue;
            }
            zeros = if b == 0 { zeros + 1 } else { 0 };
            data.push(b);
        }
        Self { data, pos: 0 }
    }
    fn bit(&mut self) -> Result<u32> {
        let byte = self.data.get(self.pos / 8).ok_or_else(|| MediaError::parse(None, "SPS truncated"))?;
        let b = (byte >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Ok(u32::from(b))
    }
    fn bits(&mut self, n: u32) -> Result<u32> {
        let mut v = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit()?;
        }
        Ok(v)
    }
    fn ue(&mut self) -> Result<u32> {
        let mut zeros = 0;
        while self.bit()? == 0 {
            zeros += 1;
            if zeros > 31 {
                return Err(MediaError::parse(None, "bad exp-golomb"));
            }
        }
        Ok(((1u64 << zeros) - 1 + u64::from(self.bits(zeros)?)) as u32)
    }
    fn se(&mut self) -> Result<i32> {
        let k = self.ue()?;
        Ok(if k % 2 == 1 { k.div_ceil(2) as i32 } else { -((k / 2) as i32) })
    }
}

fn skip_scaling_list(b: &mut Bits, size: usize) -> Result<()> {
    let mut last = 8i32;
    let mut next = 8i32;
    for _ in 0..size {
        if next != 0 {
            next = (last + b.se()? + 256) % 256;
        }
        last = if next == 0 { last } else { next };
    }
    Ok(())
}

/// Parse an SPS NAL unit (with its 1-byte header).
pub fn parse_sps(nal_unit: &[u8]) -> Result<SpsInfo> {
    if nal_type(nal_unit) != nal::SPS || nal_unit.len() < 4 {
        return Err(MediaError::parse(None, "not an SPS"));
    }
    let profile_idc = nal_unit[1];
    let constraint_flags = nal_unit[2];
    let level_idc = nal_unit[3];
    let mut b = Bits::new(&nal_unit[4..]);
    b.ue()?; // seq_parameter_set_id
    let mut chroma_format_idc = 1;
    if matches!(profile_idc, 100 | 110 | 122 | 244 | 44 | 83 | 86 | 118 | 128 | 138 | 139 | 134 | 135) {
        chroma_format_idc = b.ue()?;
        if chroma_format_idc == 3 {
            b.bit()?; // separate_colour_plane_flag
        }
        b.ue()?; // bit_depth_luma_minus8
        b.ue()?; // bit_depth_chroma_minus8
        b.bit()?; // qpprime_y_zero_transform_bypass_flag
        if b.bit()? == 1 {
            let n = if chroma_format_idc == 3 { 12 } else { 8 };
            for i in 0..n {
                if b.bit()? == 1 {
                    skip_scaling_list(&mut b, if i < 6 { 16 } else { 64 })?;
                }
            }
        }
    }
    b.ue()?; // log2_max_frame_num_minus4
    match b.ue()? {
        0 => {
            b.ue()?;
        }
        1 => {
            b.bit()?;
            b.se()?;
            b.se()?;
            let n = b.ue()?;
            for _ in 0..n {
                b.se()?;
            }
        }
        _ => {}
    }
    b.ue()?; // max_num_ref_frames
    b.bit()?; // gaps_in_frame_num_value_allowed_flag
    let w_mbs = b.ue()? + 1;
    let h_map = b.ue()? + 1;
    let frame_mbs_only = b.bit()?;
    if frame_mbs_only == 0 {
        b.bit()?;
    }
    b.bit()?; // direct_8x8_inference_flag
    let (mut cl, mut cr, mut ct, mut cb) = (0, 0, 0, 0);
    if b.bit()? == 1 {
        cl = b.ue()?;
        cr = b.ue()?;
        ct = b.ue()?;
        cb = b.ue()?;
    }
    let (cux, cuy) = match chroma_format_idc {
        0 => (1, 2 - frame_mbs_only),
        1 => (2, 2 * (2 - frame_mbs_only)),
        2 => (2, 2 - frame_mbs_only),
        _ => (1, 2 - frame_mbs_only),
    };
    let width = w_mbs * 16 - cux * (cl + cr);
    let height = (2 - frame_mbs_only) * h_map * 16 - cuy * (ct + cb);
    Ok(SpsInfo { profile_idc, constraint_flags, level_idc, width, height })
}

/// The first SPS in an Annex-B stream.
pub fn find_sps(annexb: &[u8]) -> Option<SpsInfo> {
    split_annexb(annexb).into_iter().find(|n| nal_type(n) == nal::SPS).and_then(|n| parse_sps(n).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_table() {
        // 1344x768 is 84x48 = 4032 MBs: over 3.1's 3600, fits 3.2.
        assert!(!H264Level::L3_1.fits(1344, 768, 24));
        assert!(H264Level::L3_2.fits(1344, 768, 24));
        assert!(H264Level::L4_0.fits(1344, 768, 24));
        assert_eq!(H264Level::min_for(1344, 768, 24), Some(H264Level::L3_2));
        assert_eq!(H264Level::min_for(768, 1344, 24), Some(H264Level::L3_2));
        // 832x480 (Wan/480p) fits 3.1 at 16 and 24 fps.
        assert_eq!(H264Level::min_for(832, 480, 16), Some(H264Level::L3_0));
        assert_eq!(H264Level::min_for(832, 480, 30), Some(H264Level::L3_1));
        assert_eq!(H264Level::min_for(1280, 720, 30), Some(H264Level::L3_1));
        assert_eq!(H264Level::min_for(1920, 1088, 24), Some(H264Level::L4_0));
        assert_eq!(H264Level::L3_1.cb_profile_level_id(), "42e01f");
        assert_eq!(H264Level::L3_2.cb_profile_level_id(), "42e020");
        assert_eq!("3.2".parse::<H264Level>().unwrap(), H264Level::L3_2);
        assert_eq!("4".parse::<H264Level>().unwrap(), H264Level::L4_0);
        assert_eq!("L4_1".parse::<H264Level>().unwrap(), H264Level::L4_1);
        assert!("9.9".parse::<H264Level>().is_err());
        assert_eq!(H264Level::L5_2.limits().0, 2_073_600);
        assert_eq!(H264Level::L5_1.limits().0, 983_040);
    }

    #[test]
    fn annexb_split_and_avcc() {
        let s = [0, 0, 0, 1, 0x67, 1, 2, 0, 0, 1, 0x68, 3, 0, 0, 0, 1, 0x65, 9, 9];
        let n = split_annexb(&s);
        assert_eq!(n, vec![&[0x67u8, 1, 2][..], &[0x68, 3][..], &[0x65, 9, 9][..]]);
        assert!(is_idr(&s));
        let avcc = annexb_to_avcc(&s);
        assert_eq!(&avcc[..7], &[0, 0, 0, 3, 0x67, 1, 2]);
    }

    #[test]
    fn parse_sps_x264_baseline_1344x768() {
        // SPS from libx264 -profile baseline -level 3.2, 1344x768.
        // profile 66, flags 0xc0, level 32; w_mbs-1 = 83, h_map-1 = 47.
        let mut w = BitWriter::default();
        w.ue(0); // sps id
        w.ue(0); // log2_max_frame_num_minus4
        w.ue(2); // poc type 2
        w.ue(1); // max_num_ref_frames
        w.bit(0);
        w.ue(83);
        w.ue(47);
        w.bit(1); // frame_mbs_only
        w.bit(1); // direct_8x8
        w.bit(0); // no cropping
        w.bit(0); // vui
        w.bit(1); // rbsp stop bit
        let mut nal = vec![0x67, 66, 0xc0, 32];
        nal.extend(w.finish());
        let sps = parse_sps(&nal).unwrap();
        assert_eq!((sps.width, sps.height), (1344, 768));
        assert!(sps.is_constrained_baseline());
        assert_eq!(sps.profile_level_id(), "42c020");
    }

    #[test]
    fn parse_sps_with_cropping() {
        // 1920x1080: 120x68 MBs with 8 rows cropped (crop_bottom = 4).
        let mut w = BitWriter::default();
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(0); // log2_max_poc_lsb_minus4
        w.ue(1);
        w.bit(0);
        w.ue(119);
        w.ue(67);
        w.bit(1);
        w.bit(1);
        w.bit(1);
        w.ue(0);
        w.ue(0);
        w.ue(0);
        w.ue(4);
        w.bit(0);
        w.bit(1);
        let mut nal = vec![0x67, 77, 0x40, 40];
        nal.extend(w.finish());
        let sps = parse_sps(&nal).unwrap();
        assert_eq!((sps.width, sps.height), (1920, 1080));
        assert!(!sps.is_constrained_baseline());
    }

    #[derive(Default)]
    struct BitWriter {
        bits: Vec<u8>,
    }
    impl BitWriter {
        fn bit(&mut self, b: u8) {
            self.bits.push(b);
        }
        fn ue(&mut self, v: u32) {
            let x = u64::from(v) + 1;
            let n = 64 - x.leading_zeros();
            for _ in 0..n - 1 {
                self.bit(0);
            }
            for i in (0..n).rev() {
                self.bit(((x >> i) & 1) as u8);
            }
        }
        fn finish(self) -> Vec<u8> {
            let mut out = Vec::new();
            for c in self.bits.chunks(8) {
                let mut byte = 0u8;
                for (i, &b) in c.iter().enumerate() {
                    byte |= b << (7 - i);
                }
                out.push(byte);
            }
            out
        }
    }
}
