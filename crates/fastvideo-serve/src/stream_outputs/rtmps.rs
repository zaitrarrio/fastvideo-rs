//! RTMPS with a verified server certificate (docs/serve/stream-outputs.md
//! "RTMPS TLS").
//!
//! ffmpeg's `rtmps://` opens its TLS layer as a nested protocol that
//! verifies nothing by default, so a man-in-the-middle could read the
//! stream key. Whether the image's FFmpeg 4.4 passes `-tls_verify` /
//! `-ca_file` down to that nested layer, and whether its OpenSSL backend
//! then checks the host name, depends on the FFmpeg version and was not
//! something to rely on. fastvideo therefore does the TLS itself, the same
//! way for every ffmpeg build: a [`RtmpsRelay`]
//! listens on loopback, ffmpeg publishes plain RTMP to it (announcing the
//! real `tcUrl`), and every connection is carried to the server over a
//! rustls session that
//!
//! - verifies the certificate chain against the system roots (the image's
//!   `ca-certificates`; `SSL_CERT_FILE` / `SSL_CERT_DIR` honoured) plus
//!   `[outputs] rtmps_ca_file`,
//! - checks the certificate names the URL's host (SNI sent),
//! - connects only to the addresses resolved and checked when the output
//!   started (no second DNS lookup).
//!
//! A verification failure is fatal: the relay refuses every later
//! connection, never falls back to an unverified session, and the
//! publisher fails the output with [`RtmpsRelay::fatal`]'s message.
//! `insecure_skip_verify` (tests against a self-signed MediaMTX, only with
//! `[outputs] allow_private`) is the one way to skip the check.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, ServerName, UnixTime};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use tokio_rustls::rustls::crypto::CryptoProvider;
use tokio_rustls::rustls::{self, ClientConfig, DigitallySignedStruct, RootCertStore, SignatureScheme};
use tokio_rustls::TlsConnector;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

fn provider() -> Arc<CryptoProvider> {
    Arc::new(rustls::crypto::aws_lc_rs::default_provider())
}

/// The trust anchors: the system's CA certificates plus `extra_ca` (PEM).
pub fn root_store(extra_ca: Option<&Path>) -> Result<RootCertStore, String> {
    let mut roots = RootCertStore::empty();
    let native = rustls_native_certs::load_native_certs();
    let (added, _ignored) = roots.add_parsable_certificates(native.certs);
    if let Some(p) = extra_ca {
        let certs = CertificateDer::pem_file_iter(p)
            .map_err(|e| format!("rtmps_ca_file {}: {e}", p.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("rtmps_ca_file {}: {e}", p.display()))?;
        if certs.is_empty() {
            return Err(format!("rtmps_ca_file {}: no PEM certificate in it", p.display()));
        }
        for c in certs {
            roots.add(c).map_err(|e| format!("rtmps_ca_file {}: {e}", p.display()))?;
        }
    }
    if roots.is_empty() {
        return Err(format!(
            "RTMPS needs CA certificates to verify the server: none found on this host ({added} loaded; install ca-certificates, or set SSL_CERT_FILE or [outputs] rtmps_ca_file)"
        ));
    }
    Ok(roots)
}

/// The TLS client: verifying (the default), or, for `insecure_skip_verify`,
/// accepting any certificate (signatures still checked).
pub fn client_config(verify: bool, extra_ca: Option<&Path>) -> Result<Arc<ClientConfig>, String> {
    let p = provider();
    let builder = ClientConfig::builder_with_provider(p.clone()).with_safe_default_protocol_versions().map_err(|e| e.to_string())?;
    let cfg = if verify {
        builder.with_root_certificates(root_store(extra_ca)?).with_no_client_auth()
    } else {
        builder.dangerous().with_custom_certificate_verifier(Arc::new(SkipVerify(p))).with_no_client_auth()
    };
    Ok(Arc::new(cfg))
}

/// `insecure_skip_verify`: any certificate, any name.
#[derive(Debug)]
struct SkipVerify(Arc<CryptoProvider>);

impl ServerCertVerifier for SkipVerify {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn verify_tls13_signature(&self, message: &[u8], cert: &CertificateDer<'_>, dss: &DigitallySignedStruct) -> Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, &self.0.signature_verification_algorithms)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}

/// Where the relay connects: the URL's host (certificate name and SNI), its
/// port and the addresses it resolved to when the output started.
#[derive(Clone, Debug)]
pub struct TlsTarget {
    pub host: String,
    pub port: u16,
    pub addrs: Vec<SocketAddr>,
}

#[derive(Debug)]
enum ConnectError {
    /// The server's certificate failed verification (fatal).
    Verify(String),
    /// Anything else (refused, timed out, reset): ffmpeg's restart retries.
    Other(String),
}

/// Whether a TLS handshake error is a certificate verification failure.
fn verify_failure(e: &std::io::Error) -> Option<String> {
    let r = e.get_ref()?.downcast_ref::<rustls::Error>()?;
    match r {
        rustls::Error::InvalidCertificate(c) => Some(format!("{c:?}")),
        rustls::Error::NoCertificatesPresented => Some("the server presented no certificate".into()),
        _ => None,
    }
}

async fn connect(connector: &TlsConnector, t: &TlsTarget) -> Result<tokio_rustls::client::TlsStream<TcpStream>, ConnectError> {
    let name = ServerName::try_from(t.host.clone()).map_err(|e| ConnectError::Other(format!("server name {}: {e}", t.host)))?;
    let mut last = format!("{} resolved to no address", t.host);
    for a in &t.addrs {
        let tcp = match tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(a)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                last = format!("connect: {e}");
                continue;
            }
            Err(_) => {
                last = "connect timed out".into();
                continue;
            }
        };
        let _ = tcp.set_nodelay(true);
        return match tokio::time::timeout(CONNECT_TIMEOUT, connector.connect(name, tcp)).await {
            Ok(Ok(tls)) => Ok(tls),
            Ok(Err(e)) => Err(match verify_failure(&e) {
                Some(v) => ConnectError::Verify(v),
                None => ConnectError::Other(format!("TLS handshake: {e}")),
            }),
            Err(_) => Err(ConnectError::Other("TLS handshake timed out".into())),
        };
    }
    Err(ConnectError::Other(last))
}

#[derive(Default)]
struct RelayState {
    fatal: Option<String>,
    last_error: Option<String>,
    sessions: u64,
}

/// A loopback listener that carries each connection to [`TlsTarget`] over a
/// verified TLS session. Dropping it stops accepting.
pub struct RtmpsRelay {
    port: u16,
    state: Arc<Mutex<RelayState>>,
    task: tokio::task::JoinHandle<()>,
}

impl RtmpsRelay {
    pub async fn start(target: TlsTarget, config: Arc<ClientConfig>) -> std::io::Result<Self> {
        // Checked here so a bad name fails the output at once.
        ServerName::try_from(target.host.clone()).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let port = listener.local_addr()?.port();
        let state: Arc<Mutex<RelayState>> = Arc::default();
        let st = state.clone();
        let connector = TlsConnector::from(config);
        let target = Arc::new(target);
        let task = tokio::spawn(async move {
            while let Ok((mut inbound, peer)) = listener.accept().await {
                if !peer.ip().is_loopback() {
                    continue;
                }
                if st.lock().unwrap_or_else(|p| p.into_inner()).fatal.is_some() {
                    // Never again after a failed verification.
                    continue;
                }
                let (connector, target, st) = (connector.clone(), target.clone(), st.clone());
                tokio::spawn(async move {
                    let _ = inbound.set_nodelay(true);
                    match connect(&connector, &target).await {
                        Ok(mut tls) => {
                            st.lock().unwrap_or_else(|p| p.into_inner()).sessions += 1;
                            let _ = tokio::io::copy_bidirectional(&mut inbound, &mut tls).await;
                        }
                        Err(ConnectError::Verify(e)) => {
                            let msg = format!(
                                "RTMPS: the TLS certificate of {}:{} failed verification ({e}); the output stops rather than connect unverified",
                                target.host, target.port
                            );
                            tracing::warn!(error = %msg, "rtmps relay");
                            st.lock().unwrap_or_else(|p| p.into_inner()).fatal = Some(msg);
                        }
                        Err(ConnectError::Other(e)) => {
                            let msg = format!("RTMPS: {}:{}: {e}", target.host, target.port);
                            tracing::warn!(error = %msg, "rtmps relay");
                            st.lock().unwrap_or_else(|p| p.into_inner()).last_error = Some(msg);
                        }
                    }
                });
            }
        });
        Ok(Self { port, state, task })
    }

    /// The loopback port ffmpeg publishes plain RTMP to.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Set once the server's certificate failed verification: the output
    /// must fail with this message.
    pub fn fatal(&self) -> Option<String> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).fatal.clone()
    }

    /// The last non-fatal connection error (refused, timed out).
    pub fn last_error(&self) -> Option<String> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).last_error.clone()
    }

    /// Verified TLS sessions opened so far.
    pub fn sessions(&self) -> u64 {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).sessions
    }
}

impl Drop for RtmpsRelay {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn pem(der: &[u8]) -> String {
        use base64::Engine as _;
        let b = base64::engine::general_purpose::STANDARD.encode(der);
        let lines: Vec<&str> = b.as_bytes().chunks(64).map(|c| std::str::from_utf8(c).unwrap()).collect();
        format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n", lines.join("\n"))
    }

    /// A TLS echo server with a certificate for `names`, issued by a fresh
    /// CA: (address, CA PEM file).
    async fn tls_echo(names: &[&str], dir: &Path) -> (SocketAddr, std::path::PathBuf) {
        let ca_key = rcgen::KeyPair::generate().unwrap();
        let mut ca_params = rcgen::CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
        ca_params.distinguished_name.push(rcgen::DnType::CommonName, "fv test CA");
        let ca = ca_params.self_signed(&ca_key).unwrap();
        let key = rcgen::KeyPair::generate().unwrap();
        let params = rcgen::CertificateParams::new(names.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
        let issuer = rcgen::Issuer::new(ca_params, ca_key);
        let leaf = params.signed_by(&key, &issuer).unwrap();
        let ca_file = dir.join("ca.pem");
        std::fs::write(&ca_file, pem(ca.der())).unwrap();
        let server = rustls::ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                vec![leaf.der().clone()],
                rustls_pki_types::PrivateKeyDer::Pkcs8(key.serialize_der().into()),
            )
            .unwrap();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server));
        let l = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((s, _)) = l.accept().await {
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    if let Ok(mut tls) = acceptor.accept(s).await {
                        let mut buf = [0u8; 64];
                        while let Ok(n) = tls.read(&mut buf).await {
                            if n == 0 || tls.write_all(&buf[..n]).await.is_err() {
                                break;
                            }
                            let _ = tls.flush().await;
                        }
                    }
                });
            }
        });
        (addr, ca_file)
    }

    async fn echo_through(relay: &RtmpsRelay) -> Option<Vec<u8>> {
        let mut c = TcpStream::connect(("127.0.0.1", relay.port())).await.ok()?;
        c.write_all(b"rtmp-handshake").await.ok()?;
        let mut buf = vec![0u8; 14];
        match tokio::time::timeout(Duration::from_secs(5), c.read_exact(&mut buf)).await {
            Ok(Ok(_)) => Some(buf),
            _ => None,
        }
    }

    async fn wait_fatal(relay: &RtmpsRelay) -> Option<String> {
        for _ in 0..100 {
            if let Some(f) = relay.fatal() {
                return Some(f);
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    #[tokio::test]
    async fn verifies_the_chain_and_the_name_and_never_falls_back() {
        let dir = tempfile::tempdir().unwrap();
        let (addr, ca) = tls_echo(&["localhost", "127.0.0.1"], dir.path()).await;
        let target = |host: &str| TlsTarget { host: host.into(), port: addr.port(), addrs: vec![addr] };

        // Trusted CA, matching name: bytes flow both ways.
        let relay = RtmpsRelay::start(target("localhost"), client_config(true, Some(&ca)).unwrap()).await.unwrap();
        assert_eq!(echo_through(&relay).await.as_deref(), Some(&b"rtmp-handshake"[..]));
        assert_eq!(relay.sessions(), 1);
        assert!(relay.fatal().is_none());
        // An IP address is checked against the certificate's IP names.
        let relay = RtmpsRelay::start(target("127.0.0.1"), client_config(true, Some(&ca)).unwrap()).await.unwrap();
        assert!(echo_through(&relay).await.is_some());

        // The system roots alone do not know the test CA: refused, fatal,
        // and the next connection is refused too (no unverified fallback).
        let relay = RtmpsRelay::start(target("localhost"), client_config(true, None).unwrap()).await.unwrap();
        assert!(echo_through(&relay).await.is_none());
        let f = wait_fatal(&relay).await.expect("a verification failure is fatal");
        assert!(f.contains("failed verification") && f.contains("UnknownIssuer"), "{f}");
        assert!(echo_through(&relay).await.is_none());
        assert_eq!(relay.sessions(), 0);

        // Trusted CA but another name: refused.
        let relay = RtmpsRelay::start(target("live.example.com"), client_config(true, Some(&ca)).unwrap()).await.unwrap();
        assert!(echo_through(&relay).await.is_none());
        let f = wait_fatal(&relay).await.expect("a name mismatch is fatal");
        assert!(f.contains("NotValidForName"), "{f}");

        // insecure_skip_verify: any certificate (tests only; spec.rs gates it).
        let relay = RtmpsRelay::start(target("live.example.com"), client_config(false, None).unwrap()).await.unwrap();
        assert!(echo_through(&relay).await.is_some());
    }

    #[tokio::test]
    async fn a_refused_connection_is_not_fatal() {
        let port = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap().local_addr().unwrap().port();
        let addr: SocketAddr = ([127, 0, 0, 1], port).into();
        let relay = RtmpsRelay::start(TlsTarget { host: "localhost".into(), port, addrs: vec![addr] }, client_config(false, None).unwrap())
            .await
            .unwrap();
        assert!(echo_through(&relay).await.is_none());
        for _ in 0..100 {
            if relay.last_error().is_some() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(relay.last_error().unwrap().contains("connect"), "{:?}", relay.last_error());
        assert!(relay.fatal().is_none());
    }

    #[test]
    fn a_bad_ca_file_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("empty.pem");
        std::fs::write(&p, "not a certificate").unwrap();
        assert!(client_config(true, Some(&p)).unwrap_err().contains("no PEM certificate"));
        assert!(client_config(true, Some(&dir.path().join("missing.pem"))).is_err());
    }
}
