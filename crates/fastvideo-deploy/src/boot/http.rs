//! [`Http`] over reqwest (rustls), on a current-thread tokio runtime of its
//! own: the boot is sequential, so each call blocks on its request.

use std::io::Write;
use std::path::Path;
use std::time::Duration;

use super::sys::{Http, Method, Request, Response, SysError};

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
            let mut f = std::fs::File::create(dest).map_err(|e| SysError::io(dest, e))?;
            let mut n = 0u64;
            while let Some(chunk) = r.chunk().await.map_err(|e| err(url, e))? {
                f.write_all(&chunk).map_err(|e| SysError::io(dest, e))?;
                n += chunk.len() as u64;
            }
            f.flush().map_err(|e| SysError::io(dest, e))?;
            Ok(n)
        })
    }
}
