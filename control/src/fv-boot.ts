// fv-boot (docs/adr/0014-fv-boot.md): the serve images' one boot sequence, a Rust binary beside fv-serve
// (crates/fastvideo-deploy/src/boot). It reads everything from the env fv-control already sends (driver check, config
// materialization, the weights plan, the upsampler sidecar, the tunnel and phase reports, the watchdog, exec fv-serve).
//
// Staging (ADR-0014 §6): images already deployed have no fv-boot, and fv-control is deployed separately from the
// images. So every shell boot fv-control still sends (cluster/payloads.ts WORKER_BOOT, EDGE_WORKER_BOOT,
// WATCHDOG_WORKER_BOOT; providers.ts PROVIDER_BOOT; serverless/payloads.ts SLS_BOOT, SLS_R2_BOOT, SLS_HUB_BOOT) starts
// with PREFER_FV_BOOT_SH: an image that carries fv-boot runs it, an older image runs the rest of the script as before.
// FV_BOOT_LEGACY=1 (any env layer) forces the shell boot on a new image too (the rollback switch; fv-boot itself honours
// it where the image entrypoint runs, with fv-entry.sh's steps only). Once every image a
// pool, endpoint or channel can run carries fv-boot, the shell bodies are deleted and fv-control sends env only.

/** fv-boot's path in the serve images (crates/fastvideo-deploy/fixtures/boot-protocol.json `fv_boot_bin`). */
export const FV_BOOT_BIN = "/opt/fastvideo-rs/bin/fv-boot";

/** The first line of every shell boot (POSIX sh: the serverless boot runs under /bin/sh). */
export const PREFER_FV_BOOT_SH = `if [ "\${FV_BOOT_LEGACY:-0}" != 1 ] && [ -x ${FV_BOOT_BIN} ]; then exec ${FV_BOOT_BIN}; fi
`;
