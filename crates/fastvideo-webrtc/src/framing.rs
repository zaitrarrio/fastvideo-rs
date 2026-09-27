//! RFC 4571 framing for ICE-TCP (RFC 6544 §3): every STUN/DTLS/RTP packet
//! on a TCP connection is prefixed by a 16-bit big-endian length.

/// Largest packet we accept on an ICE-TCP stream. WebRTC packets are at most
/// ~1500 bytes; anything much larger means the stream is not ICE-TCP.
pub const MAX_FRAME: usize = 8 * 1024;

/// Prefix `packet` with its length. `None` if it can't be framed.
pub fn frame(packet: &[u8]) -> Option<Vec<u8>> {
    let len = u16::try_from(packet.len()).ok()?;
    let mut v = Vec::with_capacity(packet.len() + 2);
    v.extend_from_slice(&len.to_be_bytes());
    v.extend_from_slice(packet);
    Some(v)
}

/// Incremental decoder for a byte stream of RFC 4571 frames.
#[derive(Debug, Default)]
pub struct Decoder {
    buf: Vec<u8>,
}

/// The stream is not RFC 4571 framed (or is hostile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("ice-tcp frame of {0} bytes exceeds the {MAX_FRAME}-byte limit")]
pub struct FrameTooLarge(pub usize);

impl Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed bytes; returns every complete frame.
    pub fn push(&mut self, bytes: &[u8]) -> Result<Vec<Vec<u8>>, FrameTooLarge> {
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        let mut at = 0;
        while self.buf.len() - at >= 2 {
            let len = u16::from_be_bytes([self.buf[at], self.buf[at + 1]]) as usize;
            if len > MAX_FRAME {
                return Err(FrameTooLarge(len));
            }
            if self.buf.len() - at - 2 < len {
                break;
            }
            out.push(self.buf[at + 2..at + 2 + len].to_vec());
            at += 2 + len;
        }
        self.buf.drain(..at);
        Ok(out)
    }

    /// Bytes buffered waiting for the rest of a frame.
    pub fn pending(&self) -> usize {
        self.buf.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_split_arbitrarily() {
        let packets: Vec<Vec<u8>> = vec![vec![1, 2, 3], vec![], vec![9; 1200], vec![7; 1]];
        let stream: Vec<u8> = packets.iter().flat_map(|p| frame(p).unwrap()).collect();
        for chunk in [1usize, 2, 3, 7, 500, stream.len()] {
            let mut d = Decoder::new();
            let mut got = Vec::new();
            for c in stream.chunks(chunk) {
                got.extend(d.push(c).unwrap());
            }
            assert_eq!(got, packets, "chunk size {chunk}");
            assert_eq!(d.pending(), 0);
        }
    }

    #[test]
    fn rejects_oversize_frames() {
        let mut d = Decoder::new();
        assert_eq!(d.push(&[0xff, 0xff]), Err(FrameTooLarge(65535)));
        assert!(frame(&vec![0; 70_000]).is_none());
    }
}
