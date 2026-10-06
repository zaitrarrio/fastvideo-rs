# syntax=docker/dockerfile:1.7
# Dependency images for tools-release.yml's GitHub-hosted fallback
# (docs/dev/tools-releases.md "GitHub-hosted fallback"). One image per build
# group, on the build pods' base image: the group's dependencies compiled by
# `cargo chef cook` into /target (a cached layer: only Cargo.toml/Cargo.lock
# changes rebuild it), plus the oxide cubins in /vol/release-cache (cached by
# the kernel crate and cutile-rs). The job then runs the real build
# (scripts/ci/tools-release.sh build-sets) in a container of this image with
# the checkout mounted, so only the workspace's own crates compile.
#
#   docker buildx build -f docker/tools-deps.Dockerfile --target deps \
#     --build-arg BASE_IMAGE=$(scripts/dev/build-base-tag.sh --image) \
#     --build-arg GROUP=serve-cuda --build-arg OXIDE_STAGE=deps-oxide .
ARG BASE_IMAGE
# deps-oxide for the groups that embed the cubins (oxide, serve-cuda,
# serve-cpu, gpucheck): their image carries the cached oxide build.
ARG OXIDE_STAGE=deps-nooxide

FROM ${BASE_IMAGE} AS base
# The same settings in every stage and in the job's container (layer cache
# keys and cargo fingerprints both see them).
ENV CARGO_TARGET_DIR=/target \
    FV_BUILD_VOLUME_DIR=/vol \
    CARGO_INCREMENTAL=0 \
    CARGO_TERM_COLOR=never \
    RUSTC_WRAPPER=/usr/local/bin/sccache \
    SCCACHE_DIR=/tmp/sccache
RUN cargo install cargo-chef --locked --version "^0.1" \
 && cargo chef --version

# The recipe: every manifest and the lock file, sources stubbed out.
FROM base AS planner
WORKDIR /src
COPY . .
RUN cargo chef prepare --recipe-path /recipe.json

# Oxide (Tile-IR NVFP4 cubins, release-artifacts-pod.sh's build_oxide): only
# its inputs are copied, so this layer is reused until they change.
FROM base AS oxide
WORKDIR /src
COPY crates/fastvideo-oxide-kernels crates/fastvideo-oxide-kernels
COPY third_party/cutile-rs third_party/cutile-rs
COPY scripts/dev/release-artifacts-pod.sh scripts/dev/release-artifacts-pod.sh
RUN FV_REL_SHA=0000000000000000000000000000000000000000 FV_GIT_SHA=deps FV_BUILD_ID=deps FV_REL_SETS=oxide \
      bash scripts/dev/release-artifacts-pod.sh \
 && ls /vol/release-cache/oxide/ \
 && rm -rf /target/release-artifacts /target/cuda-overlay \
 && (sccache --stop-server >/dev/null 2>&1 || true) && rm -rf /tmp/sccache

FROM base AS deps-nooxide
FROM base AS deps-oxide
COPY --from=oxide /vol /vol

FROM ${OXIDE_STAGE} AS deps
ARG GROUP
COPY --from=planner /recipe.json /recipe.json
COPY scripts/ci/tools-deps.sh /usr/local/bin/tools-deps.sh
RUN bash /usr/local/bin/tools-deps.sh "$GROUP" /recipe.json \
 && (sccache --stop-server >/dev/null 2>&1 || true) && rm -rf /tmp/sccache
