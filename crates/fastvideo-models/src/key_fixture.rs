//! Expected checkpoint keys and the recorded Hub headers they are checked
//! against.
//!
//! A port lists every tensor its loader reads, with the shape it expects,
//! as a function of the config ([`KeySpec`]). The unit tests compare that list
//! for the full-size config with a header dump of the real Hub checkpoint
//! (`fixtures/*.tsv`: `key<TAB>dtype<TAB>d0xd1x...`, fetched with HTTP range
//! requests, no weights), and the device crate checks that its loader asks
//! for exactly the list on a tiny config.

use std::collections::BTreeMap;

/// `key → shape`.
pub type KeySpec = BTreeMap<String, Vec<usize>>;

/// Parse a header dump. Lines are `key<TAB>dtype<TAB>shape`; `shape` is
/// `x`-separated (`scalar` for a 0-d tensor).
pub fn parse_fixture(text: &str) -> Result<BTreeMap<String, (String, Vec<usize>)>, String> {
    let mut out = BTreeMap::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let mut cols = line.split('\t');
        let (Some(key), Some(dtype), Some(shape)) = (cols.next(), cols.next(), cols.next()) else {
            return Err(format!("fixture line {}: want 3 columns", n + 1));
        };
        let dims = if shape == "scalar" {
            Vec::new()
        } else {
            shape
                .split('x')
                .map(|d| d.parse::<usize>().map_err(|e| format!("line {}: {e}", n + 1)))
                .collect::<Result<Vec<_>, _>>()?
        };
        out.insert(key.to_string(), (dtype.to_string(), dims));
    }
    Ok(out)
}

/// Differences between what a loader reads and a checkpoint header, keys
/// limited to those `filter` accepts on the header side (e.g. only the
/// decoder half of a VAE). Empty when they agree exactly.
pub fn diff_against(
    spec: &KeySpec,
    header: &BTreeMap<String, (String, Vec<usize>)>,
    filter: impl Fn(&str) -> bool,
) -> Vec<String> {
    let mut problems = Vec::new();
    for (k, shape) in spec {
        match header.get(k) {
            None => problems.push(format!("loader reads {k} {shape:?}; not in the checkpoint")),
            Some((_, s)) if s != shape => {
                problems.push(format!("{k}: loader expects {shape:?}, checkpoint has {s:?}"))
            }
            Some(_) => {}
        }
    }
    for (k, (_, s)) in header {
        if filter(k) && !spec.contains_key(k) {
            problems.push(format!("checkpoint has {k} {s:?}; the loader never reads it"));
        }
    }
    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_diffs() {
        let h = parse_fixture("a\tF32\t2x3\nb\tBF16\t4\nc\tF32\tscalar\n").unwrap();
        assert_eq!(h["a"].1, vec![2, 3]);
        assert!(h["c"].1.is_empty());
        let mut spec = KeySpec::new();
        spec.insert("a".into(), vec![2, 3]);
        spec.insert("b".into(), vec![5]);
        let d = diff_against(&spec, &h, |k| k != "c");
        assert_eq!(d.len(), 1, "{d:?}");
        assert!(d[0].contains("b: loader expects [5]"));
        spec.insert("b".into(), vec![4]);
        assert!(diff_against(&spec, &h, |k| k != "c").is_empty());
        assert_eq!(diff_against(&spec, &h, |_| true).len(), 1);
        assert!(parse_fixture("only\tone").is_err());
    }
}
