//! Graceful shutdown on SIGTERM/SIGINT (design §6.3; the drain itself is
//! [`crate::app::drain`]).

/// Resolves on the first SIGTERM or SIGINT (Ctrl-C).
pub async fn signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!(error = %e, "SIGINT handler unavailable");
            std::future::pending::<()>().await;
        }
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(e) => {
                tracing::warn!(error = %e, "SIGTERM handler unavailable");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => tracing::info!("SIGINT"),
        _ = term => tracing::info!("SIGTERM"),
    }
}
