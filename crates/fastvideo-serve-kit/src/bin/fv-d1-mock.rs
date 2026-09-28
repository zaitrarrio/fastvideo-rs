//! `fv-d1-mock [ADDR]`: the SQLite-backed D1 HTTP API mock
//! ([`fastvideo_serve_kit::d1::mock::MockD1`]) as a process, for local
//! multi-process runs (the gateway compat mode in `tests/compat/run.sh`).
//! Prints `FV-D1-MOCK <base>` (the `FV_D1_API_BASE` value) once listening.
//! `FV_D1_MOCK_TOKEN` requires `Authorization: Bearer <token>`.

use fastvideo_serve_kit::d1::mock::MockD1;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> std::io::Result<()> {
    let addr = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:0".into());
    let mut mock = MockD1::new();
    if let Ok(t) = std::env::var("FV_D1_MOCK_TOKEN") {
        if !t.is_empty() {
            mock = mock.with_token(t);
        }
    }
    let l = tokio::net::TcpListener::bind(&addr).await?;
    println!("FV-D1-MOCK http://{}/client/v4", l.local_addr()?);
    axum::serve(l, mock.router()).await
}
