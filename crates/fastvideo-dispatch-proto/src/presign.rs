//! Short-lived upload / download URLs a family dispatcher mints for one job
//! (docs/serve/dispatch-do-family.md §7.3). Pure Rust (`hmac`, `sha2`), so the
//! Durable Object (wasm32) and native hosts share it.
//!
//! - [`S3Presign`]: AWS SigV4 query presigning (`UNSIGNED-PAYLOAD`, signed
//!   header `host`) with extra signed query parameters, for
//!   `PUT …?partNumber=n&uploadId=u` on R2's S3 endpoint (or any S3).
//!   Same algorithm as serve-kit's `S3Config::presign` (checked against it in
//!   fv-serve's tests and against the AWS example here).
//! - Edge capability URLs ([`cap_url`], [`cap_verify`]): `PUT
//!   {base}/up?k=<key>&u=<upload>&n=<part>&e=<exp ms>&s=<hex hmac>` checked by
//!   the Worker front with a secret GPU hosts never see, which then writes the
//!   part through the R2 binding. No R2 API token is needed.

use std::collections::BTreeMap;

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};

type HmacSha256 = Hmac<Sha256>;

/// S3 caps presigned URLs at 7 days.
pub const MAX_PRESIGN_S: u64 = 7 * 24 * 3600;

/// Lower-case hex.
pub fn hex(b: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push(H[(x >> 4) as usize] as char);
        s.push(H[(x & 15) as usize] as char);
    }
    s
}

/// SHA-256, hex.
pub fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn hmac(key: &[u8], msg: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(key).expect("hmac accepts any key");
    m.update(msg);
    m.finalize().into_bytes().to_vec()
}

/// RFC 3986 percent-encoding as SigV4 wants it (`/` kept in paths).
pub fn uri_encode(s: &str, keep_slash: bool) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => o.push(b as char),
            b'/' if keep_slash => o.push('/'),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

/// Percent-decoding (`+` stays `+`); `None` on a bad escape or non-UTF-8.
pub fn uri_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            let h = b.get(i + 1..i + 3)?;
            let v = u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok()?;
            out.push(v);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `a=1&b=2` → map (decoded; a repeated key keeps the last value).
pub fn parse_query(q: &str) -> BTreeMap<String, String> {
    q.trim_start_matches('?')
        .split('&')
        .filter(|p| !p.is_empty())
        .filter_map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            Some((uri_decode(k)?, uri_decode(v)?))
        })
        .collect()
}

/// Days since 1970-01-01 → (year, month, day) (proleptic Gregorian).
fn civil(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `(YYYYMMDD, YYYYMMDDTHHMMSSZ)` of a Unix time (seconds, UTC).
pub fn amz_dates(unix_s: i64) -> (String, String) {
    let days = unix_s.div_euclid(86_400);
    let secs = unix_s.rem_euclid(86_400);
    let (y, m, d) = civil(days);
    let date = format!("{y:04}{m:02}{d:02}");
    let ts = format!("{date}T{:02}{:02}{:02}Z", secs / 3600, (secs / 60) % 60, secs % 60);
    (date, ts)
}

/// An S3-compatible bucket to presign for (R2: region `auto`, endpoint
/// `https://<account>.r2.cloudflarestorage.com`, path style).
#[derive(Clone, PartialEq, Eq)]
pub struct S3Presign {
    /// `https://host[:port][/base]`.
    pub endpoint: String,
    pub region: String,
    pub bucket: String,
    pub access_key: String,
    pub secret_key: String,
    /// `https://endpoint/bucket/key` (true) vs `https://bucket.endpoint/key`.
    pub path_style: bool,
}

impl std::fmt::Debug for S3Presign {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Presign")
            .field("endpoint", &self.endpoint)
            .field("region", &self.region)
            .field("bucket", &self.bucket)
            .field("path_style", &self.path_style)
            .finish_non_exhaustive()
    }
}

impl S3Presign {
    /// `(scheme, host[:port], base path)` of the endpoint.
    fn split_endpoint(&self) -> (String, String, String) {
        let (scheme, rest) = self.endpoint.split_once("://").unwrap_or(("https", self.endpoint.as_str()));
        let (host, path) = match rest.find('/') {
            Some(i) => (&rest[..i], rest[i..].trim_end_matches('/')),
            None => (rest, ""),
        };
        (scheme.to_owned(), host.to_owned(), path.to_owned())
    }

    /// A SigV4 query-presigned URL for `method key`, with `extra` query
    /// parameters signed too (e.g. `partNumber`, `uploadId`).
    pub fn presign(&self, method: &str, key: &str, extra: &[(&str, &str)], expires_s: u64, now_unix_s: i64) -> String {
        let expires = expires_s.clamp(1, MAX_PRESIGN_S);
        let (scheme, ep_host, base) = self.split_endpoint();
        let ekey = uri_encode(key, true);
        let (host, path) = if self.path_style {
            (ep_host, format!("{base}/{}/{ekey}", uri_encode(&self.bucket, false)))
        } else {
            (format!("{}.{ep_host}", self.bucket), format!("{base}/{ekey}"))
        };
        let (date, ts) = amz_dates(now_unix_s);
        let scope = format!("{date}/{}/s3/aws4_request", self.region);
        let mut q: Vec<(String, String)> = vec![
            ("X-Amz-Algorithm".into(), "AWS4-HMAC-SHA256".into()),
            ("X-Amz-Credential".into(), format!("{}/{scope}", self.access_key)),
            ("X-Amz-Date".into(), ts.clone()),
            ("X-Amz-Expires".into(), expires.to_string()),
            ("X-Amz-SignedHeaders".into(), "host".into()),
        ];
        q.extend(extra.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())));
        let mut enc: Vec<(String, String)> = q.iter().map(|(k, v)| (uri_encode(k, false), uri_encode(v, false))).collect();
        enc.sort();
        let cq = enc.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("&");
        let creq = format!("{method}\n{path}\n{cq}\nhost:{host}\n\nhost\nUNSIGNED-PAYLOAD");
        let sts = format!("AWS4-HMAC-SHA256\n{ts}\n{scope}\n{}", sha256_hex(creq.as_bytes()));
        let k = hmac(format!("AWS4{}", self.secret_key).as_bytes(), date.as_bytes());
        let k = hmac(&k, self.region.as_bytes());
        let k = hmac(&k, b"s3");
        let k = hmac(&k, b"aws4_request");
        let sig = hex(&hmac(&k, sts.as_bytes()));
        format!("{scheme}://{host}{path}?{cq}&X-Amz-Signature={sig}")
    }

    /// `PUT` of one part of a multipart upload.
    pub fn part_url(&self, key: &str, upload_id: &str, part: u16, expires_s: u64, now_unix_s: i64) -> String {
        let n = part.to_string();
        self.presign("PUT", key, &[("partNumber", &n), ("uploadId", upload_id)], expires_s, now_unix_s)
    }
}

/// What an edge capability URL grants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cap {
    /// `PUT` part `part` of upload `upload_id` of object `key`.
    Part { key: String, upload_id: String, part: u16 },
    /// `GET` object `key`.
    Get { key: String },
}

fn cap_msg(kind: &str, fields: &[(&str, &str)]) -> String {
    let mut m = String::from(kind);
    for (k, v) in fields {
        m.push('\n');
        m.push_str(k);
        m.push('=');
        m.push_str(v);
    }
    m
}

/// A capability URL on the Worker front: `{base}/up?…` for a part,
/// `{base}/dl?…` for a download, valid until `exp_ms`.
pub fn cap_url(base: &str, secret: &[u8], cap: &Cap, exp_ms: i64) -> String {
    let base = base.trim_end_matches('/');
    let e = exp_ms.to_string();
    match cap {
        Cap::Part { key, upload_id, part } => {
            let n = part.to_string();
            let f = [("k", key.as_str()), ("u", upload_id.as_str()), ("n", n.as_str()), ("e", e.as_str())];
            let s = hex(&hmac(secret, cap_msg("up", &f).as_bytes()));
            format!("{base}/up?k={}&u={}&n={n}&e={e}&s={s}", uri_encode(key, false), uri_encode(upload_id, false))
        }
        Cap::Get { key } => {
            let f = [("k", key.as_str()), ("e", e.as_str())];
            let s = hex(&hmac(secret, cap_msg("dl", &f).as_bytes()));
            format!("{base}/dl?k={}&e={e}&s={s}", uri_encode(key, false))
        }
    }
}

/// Checks a capability URL's path (`/up` or `/dl`) and query against
/// `secret` and `now_ms` (constant-time signature check).
pub fn cap_verify(path: &str, query: &str, secret: &[u8], now_ms: i64) -> Result<Cap, &'static str> {
    let q = parse_query(query);
    let get = |k: &str| q.get(k).map(String::as_str).ok_or("a field is missing");
    let e = get("e")?;
    let exp: i64 = e.parse().map_err(|_| "bad expiry")?;
    let (kind, fields, cap): (&str, Vec<(&str, &str)>, Cap) = match path.trim_end_matches('/') {
        "/up" => {
            let (k, u, n) = (get("k")?, get("u")?, get("n")?);
            let part: u16 = n.parse().map_err(|_| "bad part number")?;
            if part == 0 || part > 10_000 {
                return Err("bad part number");
            }
            ("up", vec![("k", k), ("u", u), ("n", n), ("e", e)], Cap::Part { key: k.to_owned(), upload_id: u.to_owned(), part })
        }
        "/dl" => {
            let k = get("k")?;
            ("dl", vec![("k", k), ("e", e)], Cap::Get { key: k.to_owned() })
        }
        _ => return Err("not a capability route"),
    };
    let sig = get("s")?;
    let want = hex(&hmac(secret, cap_msg(kind, &fields).as_bytes()));
    let ok = sig.len() == want.len() && sig.bytes().zip(want.bytes()).fold(0u8, |a, (x, y)| a | (x ^ y)) == 0;
    if !ok {
        return Err("bad signature");
    }
    if exp < now_ms {
        return Err("expired");
    }
    Ok(cap)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aws() -> S3Presign {
        S3Presign {
            endpoint: "https://s3.amazonaws.com".into(),
            region: "us-east-1".into(),
            bucket: "examplebucket".into(),
            access_key: "AKIAIOSFODNN7EXAMPLE".into(),
            secret_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".into(),
            path_style: false,
        }
    }

    #[test]
    fn dates() {
        assert_eq!(amz_dates(1_369_353_600), ("20130524".into(), "20130524T000000Z".into()));
        assert_eq!(amz_dates(0), ("19700101".into(), "19700101T000000Z".into()));
        // 2024-02-29T23:59:59Z (leap day).
        assert_eq!(amz_dates(1_709_251_199), ("20240229".into(), "20240229T235959Z".into()));
        assert_eq!(amz_dates(1_790_000_000), ("20260921".into(), "20260921T141320Z".into()));
    }

    /// The AWS SigV4 documentation example ("Example: presigned URL").
    #[test]
    fn sigv4_matches_aws_example() {
        let u = aws().presign("GET", "test.txt", &[], 86_400, 1_369_353_600);
        assert!(u.starts_with("https://examplebucket.s3.amazonaws.com/test.txt?"), "{u}");
        assert!(u.ends_with("&X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404"), "{u}");
    }

    #[test]
    fn part_urls_sign_their_query_and_clamp() {
        let mut c = aws();
        c.path_style = true;
        c.endpoint = "http://127.0.0.1:9000/base/".into();
        let u = c.part_url("out/a b.mp4", "up/1+2", 3, 30 * 86_400, 1_369_353_600);
        assert!(u.starts_with("http://127.0.0.1:9000/base/examplebucket/out/a%20b.mp4?"), "{u}");
        let q = parse_query(u.split_once('?').unwrap().1);
        assert_eq!(q["partNumber"], "3");
        assert_eq!(q["uploadId"], "up/1+2");
        assert_eq!(q["X-Amz-Expires"], "604800");
        // A different part changes the signature.
        let v = c.part_url("out/a b.mp4", "up/1+2", 4, 30 * 86_400, 1_369_353_600);
        assert_ne!(parse_query(v.split_once('?').unwrap().1)["X-Amz-Signature"], q["X-Amz-Signature"]);
    }

    #[test]
    fn capability_urls() {
        let s = b"upload-secret";
        let cap = Cap::Part { key: "outputs/wan/j1/1-1/out put.mp4".into(), upload_id: "U/x=1".into(), part: 2 };
        let u = cap_url("https://edge.example/", s, &cap, 10_000);
        let (path, q) = u.strip_prefix("https://edge.example").unwrap().split_once('?').unwrap();
        assert_eq!(path, "/up");
        assert_eq!(cap_verify(path, q, s, 9_999), Ok(cap.clone()));
        assert_eq!(cap_verify(path, q, s, 10_001), Err("expired"));
        assert_eq!(cap_verify(path, q, b"other", 0), Err("bad signature"));
        let tampered = q.replace("n=2", "n=3");
        assert_eq!(cap_verify(path, &tampered, s, 0), Err("bad signature"));
        assert_eq!(cap_verify("/dl", q, s, 0), Err("bad signature"));
        let g = Cap::Get { key: "k/1".into() };
        let u = cap_url("https://edge.example", s, &g, 5);
        let (path, q) = u.strip_prefix("https://edge.example").unwrap().split_once('?').unwrap();
        assert_eq!(cap_verify(path, q, s, 1), Ok(g));
        assert_eq!(cap_verify("/x", q, s, 1), Err("not a capability route"));
    }

    #[test]
    fn query_round_trip() {
        let q = parse_query("a=1&b=%2Fx%20y&c&d=%zz");
        assert_eq!(q["a"], "1");
        assert_eq!(q["b"], "/x y");
        assert_eq!(q["c"], "");
        assert!(!q.contains_key("d"));
    }
}
