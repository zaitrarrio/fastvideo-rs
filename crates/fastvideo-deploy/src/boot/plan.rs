//! The weights rows fv-control sends (`FV_WEIGHTS_TREES_B64`, providers.ts
//! `weightsPlan`, serverless/weights.ts): base64 of tab-separated lines,
//!
//! ```text
//! tree<TAB><dest><TAB><hub repo><TAB><revision><TAB><space-separated globs>
//! aux<TAB><path under FV_WEIGHTS><TAB><url><TAB><sha256><TAB><bytes>
//! ```
//!
//! (r2 plans carry only `aux` rows: the trees come from the fetcher's plan),
//! the recorded hashes of Hub trees (`FV_HUB_SHA256_B64`,
//! `dest<TAB>path<TAB>bytes|-<TAB>sha256[|sha256]`), and the serverless cache
//! check's `FV_R2_TREE_REVS` (`dest@rev` words).

use base64::Engine as _;

use super::error::BootError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeRow {
    pub dest: String,
    pub repo: String,
    pub revision: String,
    pub globs: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuxRow {
    pub path: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Tree(TreeRow),
    Aux(AuxRow),
}

fn decode(var: &str, b64: &str) -> Result<String, BootError> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.trim())
        .map_err(|_| BootError::Env(format!("{var} does not decode")))?;
    String::from_utf8(bytes).map_err(|_| BootError::Env(format!("{var} is not UTF-8")))
}

/// A relative path that stays under its root.
pub fn safe_rel(p: &str) -> bool {
    !p.is_empty()
        && !p.starts_with('/')
        && !p.contains('\\')
        && p.split('/').all(|x| !x.is_empty() && x != "." && x != "..")
}

/// The rows of `FV_WEIGHTS_TREES_B64` (unknown kinds and blank lines are skipped, as the shell loop did).
pub fn rows(b64: Option<&str>) -> Result<Vec<Row>, BootError> {
    let Some(b64) = b64 else {
        return Ok(Vec::new());
    };
    let text = decode("FV_WEIGHTS_TREES_B64", b64)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f.first().copied() {
            Some("tree") => {
                let dest = f.get(1).copied().unwrap_or_default();
                if !safe_rel(dest) {
                    return Err(BootError::Env(format!(
                        "FV_WEIGHTS_TREES_B64: bad tree {dest:?}"
                    )));
                }
                out.push(Row::Tree(TreeRow {
                    dest: dest.to_owned(),
                    repo: f.get(2).copied().unwrap_or_default().to_owned(),
                    revision: f.get(3).copied().unwrap_or_default().to_owned(),
                    globs: f.get(4).copied().unwrap_or_default().to_owned(),
                }));
            }
            Some("aux") => {
                let path = f.get(1).copied().unwrap_or_default();
                if !safe_rel(path) {
                    return Err(BootError::Env(format!(
                        "FV_WEIGHTS_TREES_B64: bad aux path {path:?}"
                    )));
                }
                out.push(Row::Aux(AuxRow {
                    path: path.to_owned(),
                    url: f.get(2).copied().unwrap_or_default().to_owned(),
                    sha256: f.get(3).copied().unwrap_or_default().to_ascii_lowercase(),
                    bytes: f.get(4).and_then(|b| b.trim().parse().ok()).unwrap_or(0),
                }));
            }
            _ => {}
        }
    }
    Ok(out)
}

/// One recorded file of a Hub tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubSum {
    pub dest: String,
    pub path: String,
    pub bytes: Option<u64>,
    pub sha256: Vec<String>,
}

/// The rows of `FV_HUB_SHA256_B64`.
pub fn hub_sums(b64: Option<&str>) -> Result<Vec<HubSum>, BootError> {
    let Some(b64) = b64 else {
        return Ok(Vec::new());
    };
    let text = decode("FV_HUB_SHA256_B64", b64)?;
    Ok(text
        .lines()
        .filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            if f.len() < 4 || f[0].is_empty() {
                return None;
            }
            Some(HubSum {
                dest: f[0].to_owned(),
                path: f[1].to_owned(),
                bytes: if f[2] == "-" {
                    None
                } else {
                    f[2].trim().parse().ok()
                },
                sha256: f[3]
                    .split('|')
                    .map(|h| h.trim().to_ascii_lowercase())
                    .filter(|h| !h.is_empty())
                    .collect(),
            })
        })
        .collect())
}

/// `FV_R2_TREE_REVS`: `dest@rev` words.
pub fn tree_revs(s: Option<&str>) -> Vec<(String, String)> {
    s.unwrap_or_default()
        .split_whitespace()
        .filter_map(|w| {
            let (d, r) = w.rsplit_once('@')?;
            (!d.is_empty() && !r.is_empty()).then(|| (d.to_owned(), r.to_owned()))
        })
        .collect()
}

/// A glob as the shell's `case` matched tree names: `*` any run, `?` one char.
pub fn glob(pat: &str, s: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    let (mut pi, mut ti) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(sp) = star {
            pi = sp + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == '*' {
        pi += 1;
    }
    pi == p.len()
}

/// Whether any of the space- or comma-separated globs matches.
pub fn any_glob(globs: &str, s: &str) -> bool {
    globs
        .split([' ', ','])
        .filter(|g| !g.is_empty())
        .any(|g| glob(g, s))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(s: &str) -> String {
        base64::engine::general_purpose::STANDARD.encode(s)
    }

    #[test]
    fn plan_rows() {
        let t = "tree\th3-base\tFastVideo/H3\tabc123\t*.json transformer/*\naux\tauxiliary/tae/taeh3.pth\thttps://x/t\tABCD\t42\nnoise\n\n";
        let r = rows(Some(&b64(t))).unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(
            r[0],
            Row::Tree(TreeRow {
                dest: "h3-base".into(),
                repo: "FastVideo/H3".into(),
                revision: "abc123".into(),
                globs: "*.json transformer/*".into()
            })
        );
        assert_eq!(
            r[1],
            Row::Aux(AuxRow {
                path: "auxiliary/tae/taeh3.pth".into(),
                url: "https://x/t".into(),
                sha256: "abcd".into(),
                bytes: 42
            })
        );
        assert!(rows(Some(&b64("aux\t../x\tu\th\t1\n"))).is_err());
        assert!(rows(Some("not base64!")).is_err());
        assert!(rows(None).unwrap().is_empty());
    }

    #[test]
    fn sums_and_revs() {
        let s = hub_sums(Some(&b64(
            "h3-base\ta/b.safetensors\t10\tAA|bb\nh3-base\tc\t-\tcc\n",
        )))
        .unwrap();
        assert_eq!(s[0].bytes, Some(10));
        assert_eq!(s[0].sha256, vec!["aa".to_owned(), "bb".to_owned()]);
        assert_eq!(s[1].bytes, None);
        assert_eq!(
            tree_revs(Some("h3-base@r1 auxiliary/tae@r2 bad")),
            vec![
                ("h3-base".into(), "r1".into()),
                ("auxiliary/tae".into(), "r2".into())
            ]
        );
    }

    #[test]
    fn globs() {
        assert!(glob("*_mxfp8", "h3-base_mxfp8"));
        assert!(!glob("*_mxfp8", "h3-base"));
        assert!(glob("auxiliary/*", "auxiliary/tae"));
        assert!(any_glob("*_mxfp8 auxiliary/*", "auxiliary/lpips"));
        assert!(any_glob("a,b", "b"));
        assert!(!any_glob("", "x"));
    }
}
