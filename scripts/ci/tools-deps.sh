#!/usr/bin/env bash
# The dependency layer of one tools-release group (docker/tools-deps.Dockerfile,
# docs/dev/tools-releases.md "GitHub-hosted fallback"): `cargo chef cook` of
# exactly the dependencies scripts/dev/release-artifacts-pod.sh (or
# scripts/serve/check.sh) compiles for that group, with the same profile,
# features, target dir and environment, so the real build in a container of
# this image only compiles the workspace's own crates.
#
#   tools-deps.sh <group> <recipe.json>
#
# Groups: oxide, serve-cuda, serve-cpu, serve-fake, gpucheck, gpucheck-tests,
# gpucheck-vast, hf-fm (the release sets), check-lint, check-test (check.sh).
# Keep in step with release-artifacts-pod.sh and check.sh: a mismatch only
# costs time (cargo recompiles what differs), never correctness.
set -euo pipefail
group="${1:?group}" recipe="${2:?recipe.json}"
cook() { cargo chef cook --recipe-path "$recipe" "$@"; }
# release-artifacts-pod.sh's settings (the job env of the build pod).
rel() {
  CARGO_PROFILE_RELEASE_LTO=off CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_PANIC=unwind \
    CUDARC_CUDA_VERSION=13000 NVCC="$CUDA_HOME/bin/nvcc" "$@"
}
serve_pkgs=(-p fastvideo-protocol -p fastvideo-engine-service -p fastvideo-media -p fastvideo-webrtc
  -p fastvideo-serve-kit -p fastvideo-openai-videos -p fastvideo-minimax -p fastvideo-ltxapi -p fastvideo-fal
  -p fastvideo-reactor -p fastvideo-deploy -p fastvideo-autoscale -p fastvideo-serve -p fastvideo-dispatch-proto
  -p fastvideo-edge)
case "$group" in
  oxide) ;;   # its own stage (fv-oxide-aot), cached by its inputs
  serve-cuda) rel cook --release -p fastvideo-serve --features cuda,http-client ;;
  serve-cpu) rel cook --release -p fastvideo-serve --features http-client ;;
  serve-fake)
    # The recipe's `--config profile.dev.package.<C crate>.opt-level=3`, for
    # the cook and the build alike.
    mkdir -p "$CARGO_HOME"
    cat >>"$CARGO_HOME/config.toml" <<'EOF'
[profile.dev.package.openh264-sys2]
opt-level = 3
[profile.dev.package.openh264]
opt-level = 3
[profile.dev.package.audiopus_sys]
opt-level = 3
[profile.dev.package.libwebp-sys]
opt-level = 3
EOF
    CARGO_PROFILE_DEV_DEBUG=0 rel cook -p fastvideo-serve --features fake,full --bin fv-serve ;;
  gpucheck) rel cook --release -p fastvideo-gpucheck --features cuda ;;
  gpucheck-tests) rel cook --tests -p fastvideo-gpucheck -p fastvideo-cudarc ;;
  gpucheck-vast)
    PATH="$(tr ':' '\n' <<<"$PATH" | grep -v "^$CUDA_HOME/bin$" | paste -sd:)" \
      CARGO_TARGET_DIR="$CARGO_TARGET_DIR/vast" rel env NVCC=/usr/local/cuda-13.0/bin/nvcc \
      cargo chef cook --recipe-path "$recipe" --release -p fastvideo-gpucheck --features cuda ;;
  hf-fm)
    # release-artifacts-pod.sh installs into the release cache; then it only re-checks.
    env -u CARGO_TARGET_DIR cargo install hf-fetch-model --features cli --root "$FV_BUILD_VOLUME_DIR/release-cache/hf-fm-root" ;;
  check-lint)
    CARGO_PROFILE_DEV_DEBUG=0 cook --check --all-targets "${serve_pkgs[@]}" ;;
  check-test)
    CARGO_PROFILE_DEV_DEBUG=0 cook --tests "${serve_pkgs[@]}"
    CARGO_PROFILE_DEV_DEBUG=0 cook "${serve_pkgs[@]}" ;;
  *) echo "tools-deps.sh: unknown group $group" >&2; exit 2 ;;
esac
