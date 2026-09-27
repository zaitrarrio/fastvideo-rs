//! Minimal STUN binding client (RFC 8489) for the WHIP publisher's
//! server-reflexive candidate (design §5.8: "Host candidates plus a
//! STUN-derived srflx candidate are gathered before the POST").
//!
//! Only a Binding request and the (XOR-)MAPPED-ADDRESS of a success response
//! are needed; the host sends the request from its shared UDP socket and
//! routes the response back by transaction id.

use std::hash::{BuildHasher, Hasher};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};

pub const MAGIC_COOKIE: u32 = 0x2112_A442;
const BINDING_REQUEST: u16 = 0x0001;
const BINDING_SUCCESS: u16 = 0x0101;
const ATTR_MAPPED_ADDRESS: u16 = 0x0001;
const ATTR_XOR_MAPPED_ADDRESS: u16 = 0x0020;

pub type TransactionId = [u8; 12];

/// A fresh, unpredictable transaction id.
pub fn new_transaction_id() -> TransactionId {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut id = [0u8; 12];
    for (i, chunk) in id.chunks_mut(8).enumerate() {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(n);
        h.write_usize(i);
        let v = h.finish().to_le_bytes();
        chunk.copy_from_slice(&v[..chunk.len()]);
    }
    id
}

/// Encode a Binding request with no attributes.
pub fn binding_request(tx: &TransactionId) -> Vec<u8> {
    let mut v = Vec::with_capacity(20);
    v.extend_from_slice(&BINDING_REQUEST.to_be_bytes());
    v.extend_from_slice(&0u16.to_be_bytes());
    v.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    v.extend_from_slice(tx);
    v
}

/// Encode a Binding success response carrying XOR-MAPPED-ADDRESS (used by
/// tests and the in-process fake STUN server).
pub fn binding_success(tx: &TransactionId, mapped: SocketAddr) -> Vec<u8> {
    let mut attr = Vec::new();
    let xport = mapped.port() ^ (MAGIC_COOKIE >> 16) as u16;
    match mapped.ip() {
        IpAddr::V4(ip) => {
            attr.extend_from_slice(&[0, 0x01]);
            attr.extend_from_slice(&xport.to_be_bytes());
            attr.extend_from_slice(&(u32::from(ip) ^ MAGIC_COOKIE).to_be_bytes());
        }
        IpAddr::V6(ip) => {
            attr.extend_from_slice(&[0, 0x02]);
            attr.extend_from_slice(&xport.to_be_bytes());
            let mut key = [0u8; 16];
            key[..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
            key[4..].copy_from_slice(tx);
            let o = ip.octets();
            attr.extend((0..16).map(|i| o[i] ^ key[i]));
        }
    }
    let mut v = Vec::with_capacity(24 + attr.len());
    v.extend_from_slice(&BINDING_SUCCESS.to_be_bytes());
    v.extend_from_slice(&((4 + attr.len()) as u16).to_be_bytes());
    v.extend_from_slice(&MAGIC_COOKIE.to_be_bytes());
    v.extend_from_slice(tx);
    v.extend_from_slice(&ATTR_XOR_MAPPED_ADDRESS.to_be_bytes());
    v.extend_from_slice(&(attr.len() as u16).to_be_bytes());
    v.extend_from_slice(&attr);
    v
}

/// If `buf` is a STUN Binding request, its transaction id.
pub fn parse_binding_request(buf: &[u8]) -> Option<TransactionId> {
    let tx = header(buf)?;
    (u16::from_be_bytes([buf[0], buf[1]]) == BINDING_REQUEST).then_some(tx)
}

fn header(buf: &[u8]) -> Option<TransactionId> {
    if buf.len() < 20 || buf[0] & 0xc0 != 0 {
        return None;
    }
    if u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) != MAGIC_COOKIE {
        return None;
    }
    let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    if buf.len() < 20 + len {
        return None;
    }
    let mut tx = [0u8; 12];
    tx.copy_from_slice(&buf[8..20]);
    Some(tx)
}

/// Parse a Binding success response: `(transaction id, mapped address)`.
pub fn parse_binding_success(buf: &[u8]) -> Option<(TransactionId, SocketAddr)> {
    let tx = header(buf)?;
    if u16::from_be_bytes([buf[0], buf[1]]) != BINDING_SUCCESS {
        return None;
    }
    let len = u16::from_be_bytes([buf[2], buf[3]]) as usize;
    let body = &buf[20..20 + len];
    let mut at = 0;
    let mut plain = None;
    while at + 4 <= body.len() {
        let typ = u16::from_be_bytes([body[at], body[at + 1]]);
        let alen = u16::from_be_bytes([body[at + 2], body[at + 3]]) as usize;
        let Some(val) = body.get(at + 4..at + 4 + alen) else {
            break;
        };
        match typ {
            ATTR_XOR_MAPPED_ADDRESS => return decode_addr(val, Some(&tx)).map(|a| (tx, a)),
            ATTR_MAPPED_ADDRESS => plain = decode_addr(val, None),
            _ => {}
        }
        at += 4 + alen.div_ceil(4) * 4;
    }
    plain.map(|a| (tx, a))
}

fn decode_addr(v: &[u8], xor_tx: Option<&TransactionId>) -> Option<SocketAddr> {
    if v.len() < 8 {
        return None;
    }
    let mut port = u16::from_be_bytes([v[2], v[3]]);
    if xor_tx.is_some() {
        port ^= (MAGIC_COOKIE >> 16) as u16;
    }
    let ip = match v[1] {
        0x01 => {
            let mut raw = u32::from_be_bytes([v[4], v[5], v[6], v[7]]);
            if xor_tx.is_some() {
                raw ^= MAGIC_COOKIE;
            }
            IpAddr::V4(Ipv4Addr::from(raw))
        }
        0x02 if v.len() >= 20 => {
            let mut o = [0u8; 16];
            o.copy_from_slice(&v[4..20]);
            if let Some(tx) = xor_tx {
                let mut key = [0u8; 16];
                key[..4].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
                key[4..].copy_from_slice(tx);
                for (b, k) in o.iter_mut().zip(key) {
                    *b ^= k;
                }
            }
            IpAddr::V6(Ipv6Addr::from(o))
        }
        _ => return None,
    };
    Some(SocketAddr::new(ip, port))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_and_response_round_trip() {
        let tx = new_transaction_id();
        assert_ne!(tx, new_transaction_id());
        let req = binding_request(&tx);
        assert_eq!(req.len(), 20);
        assert_eq!(parse_binding_request(&req), Some(tx));
        for mapped in ["203.0.113.9:40123", "[2001:db8::7]:5000"] {
            let mapped: SocketAddr = mapped.parse().unwrap();
            let resp = binding_success(&tx, mapped);
            assert_eq!(parse_binding_success(&resp), Some((tx, mapped)));
            assert_eq!(parse_binding_request(&resp), None);
        }
    }

    #[test]
    fn rejects_non_stun() {
        assert_eq!(parse_binding_success(&[0u8; 10]), None);
        // DTLS/RTP first bytes have the top bits set.
        let mut rtp = vec![0x80u8; 40];
        rtp[4..8].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
        assert_eq!(parse_binding_success(&rtp), None);
    }

    #[test]
    fn decodes_the_rfc5769_ipv4_vector() {
        // RFC 5769 §2.2 response, attributes other than XOR-MAPPED-ADDRESS
        // replaced by a SOFTWARE attribute we skip.
        let tx: TransactionId = [
            0xb7, 0xe7, 0xa7, 0x01, 0xbc, 0x34, 0xd6, 0x86, 0xfa, 0x87, 0xdf, 0xae,
        ];
        let mut msg = vec![0x01, 0x01, 0x00, 0x14, 0x21, 0x12, 0xa4, 0x42];
        msg.extend_from_slice(&tx);
        msg.extend_from_slice(&[0x80, 0x22, 0x00, 0x04, b't', b'e', b's', b't']);
        msg.extend_from_slice(&[
            0x00, 0x20, 0x00, 0x08, 0x00, 0x01, 0xa1, 0x47, 0xe1, 0x12, 0xa6, 0x43,
        ]);
        assert_eq!(
            parse_binding_success(&msg),
            Some((tx, "192.0.2.1:32853".parse().unwrap()))
        );
    }
}
