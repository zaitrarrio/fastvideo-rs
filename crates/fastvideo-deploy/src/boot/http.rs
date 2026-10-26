//! [`Http`] over reqwest (rustls), on a current-thread tokio runtime of its
//! own: the boot is sequential, so each call blocks on its request.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use super::sys::{too_large, Http, Method, Request, Response, SysError};

pub struct ReqwestHttp {
    rt: tokio::runtime::Runtime,
    client: reqwest::Client,
}

impl ReqwestHttp {
    pub fn new() -> Result<Self, SysError> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| SysError::Http {
                url: String::new(),
                why: format!("tokio runtime: {e}"),
            })?;
        let client = reqwest::Client::builder()
            .user_agent(concat!("fv-boot/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| SysError::Http {
                url: String::new(),
                why: e.to_string(),
            })?;
        Ok(Self { rt, client })
    }
}

impl ReqwestHttp {
    /// GET `url` into `dest`, streamed; with `max`, refused on a larger
    /// `Content-Length` before any byte is written, and stopped as soon as the
    /// bytes received pass it.
    fn fetch_to(
        &self,
        url: &str,
        dest: &Path,
        timeout: Duration,
        max: Option<u64>,
    ) -> Result<u64, SysError> {
        self.rt.block_on(async {
            let mut r = self
                .client
                .get(url)
                .timeout(timeout)
                .send()
                .await
                .map_err(|e| err(url, e))?;
            let status = r.status().as_u16();
            if !(200..300).contains(&status) {
                return Err(SysError::Status {
                    url: url.to_owned(),
                    status,
                });
            }
            if let (Some(m), Some(len)) = (max, r.content_length()) {
                if len > m {
                    return Err(too_large(url, m));
                }
            }
            let mut f = std::fs::File::create(dest).map_err(|e| SysError::io(dest, e))?;
            let mut n = 0u64;
            while let Some(chunk) = r.chunk().await.map_err(|e| err(url, e))? {
                n += chunk.len() as u64;
                if let Some(m) = max {
                    if n > m {
                        return Err(too_large(url, m));
                    }
                }
                f.write_all(&chunk).map_err(|e| SysError::io(dest, e))?;
            }
            f.flush().map_err(|e| SysError::io(dest, e))?;
            Ok(n)
        })
    }
}

fn err(url: &str, e: impl std::fmt::Display) -> SysError {
    SysError::Http {
        url: url.to_owned(),
        why: e.to_string(),
    }
}

impl Http for ReqwestHttp {
    fn send(&self, req: &Request) -> Result<Response, SysError> {
        let m = match req.method {
            Method::Get => reqwest::Method::GET,
            Method::Post => reqwest::Method::POST,
            Method::Delete => reqwest::Method::DELETE,
            Method::Patch => reqwest::Method::PATCH,
            Method::Put => reqwest::Method::PUT,
        };
        let mut b = self.client.request(m, &req.url).timeout(req.timeout);
        for (k, v) in &req.headers {
            b = b.header(k, v);
        }
        if let Some(body) = &req.body {
            b = b.body(body.clone());
        }
        self.rt.block_on(async {
            let r = b.send().await.map_err(|e| err(&req.url, e))?;
            let status = r.status().as_u16();
            let body = r.bytes().await.map_err(|e| err(&req.url, e))?.to_vec();
            Ok(Response { status, body })
        })
    }

    fn download(&self, url: &str, dest: &Path, timeout: Duration) -> Result<u64, SysError> {
        self.fetch_to(url, dest, timeout, None)
    }

    fn download_capped(
        &self,
        url: &str,
        dest: &Path,
        timeout: Duration,
        max: u64,
    ) -> Result<u64, SysError> {
        let r = self.fetch_to(url, dest, timeout, Some(max));
        if r.is_err() {
            let _ = std::fs::remove_file(dest);
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;

    /// A one-request server: answers `head` then `body` (written in 4 KiB
    /// pieces, then the connection closes).
    fn serve(head: String, body: Vec<u8>) -> String {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut c, _)) = l.accept() {
                let mut buf = [0u8; 4096];
                let _ = c.read(&mut buf);
                let _ = c.write_all(head.as_bytes());
                for piece in body.chunks(4096) {
                    if c.write_all(piece).is_err() {
                        break;
                    }
                }
            }
        });
        format!("http://{addr}/f")
    }

    fn dest(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fv-http-cap-{name}-{}", std::process::id()))
    }

    #[test]
    fn a_capped_download_refuses_a_large_content_length_and_stops_mid_stream() {
        let h = ReqwestHttp::new().unwrap();
        let t = Duration::from_secs(10);
        // Within the cap: written whole.
        let url = serve(
            "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nConnection: close\r\n\r\n".into(),
            b"hello".to_vec(),
        );
        let d = dest("ok");
        assert_eq!(h.download_capped(&url, &d, t, 5).unwrap(), 5);
        assert_eq!(std::fs::read(&d).unwrap(), b"hello");
        let _ = std::fs::remove_file(&d);
        // Content-Length over the cap: refused, nothing written.
        let url = serve(
            "HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\nConnection: close\r\n\r\n".into(),
            vec![1u8; 1_000_000],
        );
        let d = dest("len");
        let e = h.download_capped(&url, &d, t, 10).unwrap_err().to_string();
        assert!(e.contains("10-byte limit"), "{e}");
        assert!(!d.exists());
        // No Content-Length: stopped once the bytes pass the cap, file removed.
        let url = serve(
            "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".into(),
            vec![2u8; 200_000],
        );
        let d = dest("stream");
        let e = h
            .download_capped(&url, &d, t, 50_000)
            .unwrap_err()
            .to_string();
        assert!(e.contains("50000-byte limit"), "{e}");
        assert!(!d.exists());
        // A non-2xx status is an error and leaves no file.
        let url = serve(
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into(),
            Vec::new(),
        );
        let d = dest("404");
        assert!(h.download_capped(&url, &d, t, 50).is_err());
        assert!(!d.exists());
    }
}
