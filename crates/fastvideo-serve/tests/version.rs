//! `fv-serve --version` prints the build identity (docs/serve/releases.md);
//! `-V` stays the bare package version.

use std::process::Command;

const DIGEST: &str = "sha256:0441e76fcff8d240eb1de5c33782ff512a3aaf90dd1fb3b0fb5bb8c5b216584c";

fn fv_serve(args: &[&str], env: &[(&str, &str)]) -> String {
    let mut c = Command::new(env!("CARGO_BIN_EXE_fv-serve"));
    c.args(args);
    for k in ["FV_VARIANT", "FV_IMAGE_REF", "FV_IMAGE_TAG", "FV_IMAGE_DIGEST", "FV_RELEASE_CHANNEL", "FV_CONFIG"] {
        c.env_remove(k);
    }
    c.envs(env.iter().copied());
    let out = c.output().expect("run fv-serve");
    assert!(out.status.success(), "{args:?}: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn long_version_has_sha_variant_and_digest() {
    let out = fv_serve(
        &["--version"],
        &[
            ("FV_VARIANT", "wan5b"),
            ("FV_IMAGE_REF", &format!("ghcr.io/zaitrarrio/fastvideo-rs-serve@{DIGEST}")),
            ("FV_IMAGE_TAG", "wan5b-sha-2cd1ba0"),
            ("FV_RELEASE_CHANNEL", "stable"),
        ],
    );
    let mut lines = out.lines();
    assert_eq!(lines.next(), Some(format!("fv-serve {}", env!("CARGO_PKG_VERSION")).as_str()), "{out}");
    let git = out.lines().find_map(|l| l.strip_prefix("git:")).expect("git line").trim();
    assert!(git == "unknown" || (git.len() == 40 && git.chars().all(|c| c.is_ascii_hexdigit())), "{out}");
    for want in ["variant:  wan5b", "channel:  stable", "tag:      wan5b-sha-2cd1ba0", &format!("digest:   {DIGEST}")] {
        assert!(out.contains(want), "missing `{want}` in:\n{out}");
    }
}

#[test]
fn short_version_is_the_package_version() {
    let out = fv_serve(&["-V"], &[]);
    assert_eq!(out.trim(), format!("fv-serve {}", env!("CARGO_PKG_VERSION")));
}

#[test]
fn without_deployment_env_only_the_build_lines() {
    let out = fv_serve(&["--version"], &[]);
    assert!(out.contains("git:"), "{out}");
    assert!(!out.contains("digest:") && !out.contains("channel:"), "{out}");
}
