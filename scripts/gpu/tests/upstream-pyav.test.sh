#!/usr/bin/env bash
# The sol-ltx25 reference's PyAV pin (docs/adr/0019-fv-lab.md §20.10), without a GPU or the stack:
#   - setup.sh pins LTX25_AV_VERSION to one exact release (no override), install_sol_ltx25 installs
#     av==$LTX25_AV_VERSION after `uv sync` (which resolves LTX-2's unpinned "av"), runs check_ltx25_av
#     before it writes its stamp, and the stamp names the pin and is v3;
#   - upstream.Dockerfile's sol-ltx25 step bind-mounts the retake fixture at the LTX25_AUDIO_FIXTURE it
#     exports, so the check runs at image build, and the fixture is in the repository;
#   - check_ltx25_av (cut from setup.sh, run in a clean shell) on a throwaway venv with a stub `av` and a
#     stub ltx_pipelines whose package __init__s raise: passes only for the pinned version and a decoded
#     [2, 222208], fails on another version, on the PyAV 19 length [2, 221152] and on a missing fixture,
#     and never runs the package __init__s.
#
#   bash scripts/gpu/tests/upstream-pyav.test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SETUP="$ROOT/scripts/gpu/upstream/setup.sh"
DOCKERFILE="$ROOT/docker/upstream.Dockerfile"
FIXTURE_REL=scripts/gpu/fixtures/beach-push-768x512-24fps.mp4
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok() { printf 'ok   %s\n' "$1"; }
bad() { printf 'FAIL %s\n' "$1"; fail=1; }

# fn <name>: one function's definition from setup.sh (its first line to the first line that is `}`).
fn() { sed -n "/^$1() {\$/,/^}\$/p" "$SETUP"; }

# 1. The pin and the installer.
pin="$(grep -E '^LTX25_AV_VERSION=' "$SETUP" || true)"
[[ "$pin" =~ ^LTX25_AV_VERSION=[0-9]+\.[0-9]+\.[0-9]+$ ]] && ok "setup.sh pins PyAV exactly ($pin)" \
  || bad "setup.sh: LTX25_AV_VERSION is not one exact release: '$pin'"
INSTALL="$(fn install_sol_ltx25)"
line() { { grep -nF -- "$1" <<<"$INSTALL" || true; } | head -1 | cut -d: -f1; }
sync_l="$(line 'uv sync -q')"; av_l="$(line '"av==$LTX25_AV_VERSION"')"
check_l="$(line 'check_ltx25_av "$py" || return 1')"; stamp_l="$(line 'echo "$stamp" >"$d/.stamp"')"
[[ -n "$sync_l" && -n "$av_l" && -n "$check_l" && -n "$stamp_l" ]] && (( sync_l < av_l && av_l < check_l && check_l < stamp_l )) \
  && ok "install_sol_ltx25: uv sync, then av==\$LTX25_AV_VERSION, then check_ltx25_av, then the stamp" \
  || bad "install_sol_ltx25 order: sync $sync_l, av $av_l, check $check_l, stamp $stamp_l"
grep -qE 'stamp="ltx2:\$LTX2_REV:sol:\$SOL_REV:av:\$LTX25_AV_VERSION:v3"' <<<"$INSTALL" \
  && ok "install_sol_ltx25's stamp names the pin and is v3" || bad "install_sol_ltx25's stamp: $(grep -o 'stamp="[^"]*"' <<<"$INSTALL")"

# 2. The image build runs the check on the fixture.
[[ -f "$ROOT/$FIXTURE_REL" ]] && ok "the retake fixture is in the repository" || bad "no $FIXTURE_REL"
stage="$(sed -n '/^FROM base AS sol-ltx25$/,/^FROM /p' "$DOCKERFILE")"
target="$(grep -oE -- "--mount=type=bind,source=$FIXTURE_REL,target=[^ ]+" <<<"$stage" | sed 's/.*target=//' || true)"
[[ -n "$target" ]] && grep -qF -- "export LTX25_AUDIO_FIXTURE=$target" <<<"$stage" \
  && grep -qE '&& \. /opt/upstream-setup/setup\.sh && install_sol_ltx25 ' <<<"$stage" \
  && ok "upstream.Dockerfile sol-ltx25: the fixture bind-mounted at $target, exported, then install_sol_ltx25" \
  || bad "upstream.Dockerfile sol-ltx25 does not mount and export the fixture for install_sol_ltx25: $stage"

# 3. check_ltx25_av on stubs.
want="${pin#LTX25_AV_VERSION=}"
V="$TMP/venv"
python3 -m venv --without-pip "$V"
SP="$("$V/bin/python" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
mkdir -p "$SP/ltx_pipelines/utils/media_io"
for i in ltx_pipelines ltx_pipelines/utils ltx_pipelines/utils/media_io; do
  printf 'raise ImportError("package __init__ executed: %s")\n' "$i" >"$SP/$i/__init__.py"
done
cat >"$SP/ltx_pipelines/utils/media_io/decode.py" <<'PY'
import os


class _Wave:
    def __init__(self, shape):
        self.shape = shape


class Audio:
    def __init__(self, shape):
        self.waveform = [_Wave(shape)]
        self.sampling_rate = 44100


def decode_audio_from_file(path, device, start_time=0.0, max_duration=None):
    assert os.path.isfile(path) and device == "cpu" and start_time == 0.0 and max_duration == 121 / 24
    return Audio((2, int(os.environ["STUB_SAMPLES"])))
PY
FX="$TMP/source.mp4"; : >"$FX"
# run <av version> <samples> [fixture]: check_ltx25_av in a clean shell, its output to $TMP/out; prints the exit.
run() {
  printf '__version__ = "%s"\n' "$1" >"$SP/av.py"
  env -i PATH="$PATH" HOME="$TMP" STUB_SAMPLES="$2" bash -c '
    set -uo pipefail
    slog() { printf "[setup] %s\n" "$*" >&2; }
    eval "$(sed -n "/^LTX25_AV_VERSION=/p" "$1")"
    eval "$(sed -n "/^check_ltx25_av() {\$/,/^}\$/p" "$1")"
    check_ltx25_av "$2" "$3"' _ "$SETUP" "$V/bin/python" "${3:-$FX}" >"$TMP/out" 2>&1 && echo 0 || echo $?
}
rc="$(run "$want" 222208)"
[[ "$rc" == 0 ]] && grep -qF "PyAV $want, decode_audio_from_file (2, 222208) at 44100 Hz: ok" "$TMP/out" \
  && ok "check_ltx25_av: PyAV $want and [2, 222208] pass" || bad "check_ltx25_av pinned + 222208: rc $rc: $(cat "$TMP/out")"
! grep -q "package __init__ executed" "$TMP/out" && ok "check_ltx25_av loads decode.py without the package __init__s" \
  || bad "check_ltx25_av ran a package __init__: $(cat "$TMP/out")"
rc="$(run 19.0.2 222208)"
[[ "$rc" != 0 ]] && grep -qF "PyAV 19.0.2, pinned $want" "$TMP/out" && ok "check_ltx25_av: another PyAV fails" \
  || bad "check_ltx25_av on PyAV 19.0.2: rc $rc: $(cat "$TMP/out")"
rc="$(run "$want" 221152)"
[[ "$rc" != 0 ]] && grep -qF "gave (2, 221152), want (2, 222208)" "$TMP/out" \
  && ok "check_ltx25_av: the PyAV 19 length [2, 221152] fails" || bad "check_ltx25_av on 221152: rc $rc: $(cat "$TMP/out")"
rc="$(run "$want" 222208 "$TMP/missing.mp4")"
[[ "$rc" != 0 ]] && grep -qF "no audio fixture at $TMP/missing.mp4" "$TMP/out" \
  && ok "check_ltx25_av: a missing fixture fails" || bad "check_ltx25_av without the fixture: rc $rc: $(cat "$TMP/out")"

exit "$fail"
