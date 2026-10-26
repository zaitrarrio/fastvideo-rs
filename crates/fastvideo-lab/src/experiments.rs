//! `fv-lab report summary --kind <experiment> <dir>`: the four experiment
//! summaries of `scripts/gpu/experiments/*-summary.py`, ported line for line
//! (docs/adr/0019-fv-lab.md §13):
//!
//! - [`h3_graphs`]: `h3-graphs-summary.py` (`cells/`, `trace/`, `hashes.tsv`);
//! - [`h3_tier_options`]: `h3-tier-options-summary.py` (`cells/`,
//!   `compare/clips`, `compare/gate`, `check/`), with `--json`;
//! - [`h3_trim_nvfp4`]: `h3-trim-nvfp4-summary.py` (`runs/`, the parity
//!   dumps), with `--json`;
//! - [`fasth3_nvfp4`]: `fasth3-nvfp4-summary.py` (`up-*`, `ours-*`, the
//!   parity dumps, the phase profile), with `--json`.
//!
//! Each reads the directory its validate script's `fetch` wrote. Where the
//! script would raise (a missing key, a value of the wrong type, a malformed
//! number), the port fails with a message instead of printing a partial
//! report. The regular expressions are matched by hand (the crate has no
//! regex engine), each with the scripts' semantics: lazy groups shortest
//! first, greedy ones longest first, as Python's backtracking tries them.

use std::path::Path;

use crate::pyfmt::{
    self, dumps, dumps_indent, exp, fixed, head_chars, ljust, median, median_py, py_str, repr,
    rjust, str_repr, tail_chars, tuple_repr, Py,
};

type R<T> = Result<T, String>;

/// A file as Python's `open(p, errors="replace")` reads it: invalid UTF-8
/// replaced, `\r\n` and `\r` read as `\n`.
fn py_text(path: &Path) -> R<String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(String::from_utf8_lossy(&b)
        .replace("\r\n", "\n")
        .replace('\r', "\n"))
}

/// Iterating a text file: each line with its `\n` (the last one may lack it).
fn py_lines(text: &str) -> Vec<&str> {
    text.split_inclusive('\n').collect()
}

/// `d.get(k)`: `None` when missing; a non-dict raises.
fn get<'a>(v: &'a Py, k: &str) -> R<&'a Py> {
    match v {
        Py::Dict(_) => Ok(v.at(k)),
        other => Err(format!("{} has no .get (looking up `{k}`)", repr(other))),
    }
}

/// `d.get(k, default)`: the default only for a missing key.
fn get_or<'a>(v: &'a Py, k: &str, default: &'a Py) -> R<&'a Py> {
    match v {
        Py::Dict(_) => Ok(v.get(k).unwrap_or(default)),
        other => Err(format!("{} has no .get (looking up `{k}`)", repr(other))),
    }
}

/// `d[k]`: a missing key raises.
fn index<'a>(v: &'a Py, k: &str) -> R<&'a Py> {
    match v {
        Py::Dict(_) => v.get(k).ok_or_else(|| format!("KeyError: {}", str_repr(k))),
        other => Err(format!("{} is not subscriptable by `{k}`", repr(other))),
    }
}

/// `x or {}`.
fn or_dict(v: &Py) -> Py {
    v.or(Py::dict())
}

/// `f"{x:.{d}f}"` of a number value; anything else raises.
fn num_f(v: &Py, d: usize) -> R<String> {
    pyfmt::fixed_py(v, d).ok_or_else(|| format!("cannot format {} as a number", repr(v)))
}

/// The scripts' `f(x, d)`: `-` for `None`, else `{x:.{d}f}`.
fn f_or_dash(v: &Py, d: usize) -> R<String> {
    if v.is_none() {
        Ok("-".into())
    } else {
        num_f(v, d)
    }
}

fn num(v: &Py) -> R<f64> {
    v.num()
        .ok_or_else(|| format!("{} is not a number", repr(v)))
}

/// Python's `float(s)` of a token (`ValueError` as an error).
fn py_float(s: &str) -> R<f64> {
    let t = s.trim();
    let lower = t.to_ascii_lowercase();
    let v = match lower.trim_start_matches(['+', '-']) {
        "nan" => Some(f64::NAN),
        "inf" | "infinity" => Some(f64::INFINITY),
        _ => None,
    };
    if let Some(v) = v {
        return Ok(if lower.starts_with('-') { -v } else { v });
    }
    t.parse::<f64>()
        .map_err(|_| format!("could not convert string to float: {}", str_repr(s)))
}

/// Sorted names of a directory (`sorted(os.listdir(d))`); `None` if it is not
/// a directory.
fn listdir(dir: &Path) -> Option<Vec<String>> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut v: Vec<String> = rd
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    Some(v)
}

/// Python's `\s` (`str.isspace`).
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python's `\w` (`str.isalnum()` or `_`).
fn py_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The ASCII digits at the start of `s` (`\d+`), if any.
fn digits_at(s: &str) -> Option<&str> {
    let n = s.bytes().take_while(u8::is_ascii_digit).count();
    (n > 0).then(|| &s[..n])
}

/// `(\{.*\})$` on what follows a fixed prefix: the rest must start with `{`
/// and end with `}`.
fn braced(rest: &str) -> Option<&str> {
    (rest.len() >= 2 && rest.starts_with('{') && rest.ends_with('}')).then_some(rest)
}

/// The ends `e` (ascending) of the prefixes of `s` that match `.+?-\d{n}`
/// (`n` digits) or, with `n == 0`, `.+?-\d+` (a maximal digit run), and are
/// followed by `next`.
fn lazy_dash_digits_ends(s: &str, n: usize, next: &str) -> Vec<usize> {
    let b = s.as_bytes();
    let mut out = Vec::new();
    for dash in 1..b.len() {
        if b[dash] != b'-' || !s.is_char_boundary(dash) {
            continue;
        }
        let run = b[dash + 1..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count();
        let end = if n == 0 {
            if run == 0 {
                continue;
            }
            dash + 1 + run
        } else {
            if run < n {
                continue;
            }
            dash + 1 + n
        };
        if s[end..].starts_with(next) {
            out.push(end);
        }
    }
    out
}

/// `re.match(r"compare-clips-(.+?-\d{n})--(.+?-\d{n})-(.+)\.json$", f)`
/// (`n == 0`: `\d+`): the three groups.
fn match_compare_clips(f: &str, n: usize) -> Option<(String, String, String)> {
    let r = f.strip_prefix("compare-clips-")?;
    for e1 in lazy_dash_digits_ends(r, n, "--") {
        let r2 = &r[e1 + 2..];
        for e2 in lazy_dash_digits_ends(r2, n, "-") {
            let r3 = &r2[e2 + 1..];
            if r3.len() > 5 && r3.ends_with(".json") && !r3.contains('\n') {
                return Some((
                    r[..e1].to_owned(),
                    r2[..e2].to_owned(),
                    r3[..r3.len() - 5].to_owned(),
                ));
            }
        }
    }
    None
}

/// `re.match(r"gate-(.+?-\d{n})--(.+?-\d{n})\.json$", f)` (`n == 0`: `\d+`).
fn match_gate(f: &str, n: usize) -> Option<(String, String)> {
    let r = f.strip_prefix("gate-")?;
    let r2_all = r.strip_suffix(".json")?;
    for e1 in lazy_dash_digits_ends(r, n, "--") {
        if e1 + 2 > r2_all.len() {
            continue;
        }
        let g2 = &r2_all[e1 + 2..];
        if lazy_dash_digits_ends(&format!("{g2}\u{0}"), n, "\u{0}").contains(&g2.len()) {
            return Some((r[..e1].to_owned(), g2.to_owned()));
        }
    }
    None
}

/// `re.match(r"compare-dumps (\S+)\s+rel_l2 (\S+)" + tail)` on a line: the
/// name, the value text and what follows the value.
fn match_compare_dumps(line: &str) -> Option<(&str, &str, &str)> {
    let r = line.strip_prefix("compare-dumps ")?;
    let n1 = r.find(py_space).unwrap_or(r.len());
    if n1 == 0 {
        return None;
    }
    let (name, rest) = r.split_at(n1);
    let ws = rest.len() - rest.trim_start_matches(py_space).len();
    if ws == 0 {
        return None;
    }
    let rest = rest[ws..].strip_prefix("rel_l2 ")?;
    let n2 = rest.find(py_space).unwrap_or(rest.len());
    if n2 == 0 {
        return None;
    }
    Some((name, &rest[..n2], &rest[n2..]))
}

/// `str.splitlines()`'s first line.
fn first_line(s: &str) -> Option<&str> {
    if s.is_empty() {
        return None;
    }
    let end = s
        .find(|c: char| {
            matches!(
                c,
                '\n' | '\r'
                    | '\u{0b}'
                    | '\u{0c}'
                    | '\u{1c}'
                    | '\u{1d}'
                    | '\u{1e}'
                    | '\u{85}'
                    | '\u{2028}'
                    | '\u{2029}'
            )
        })
        .unwrap_or(s.len());
    Some(&s[..end])
}

fn dict_of(items: Vec<(&str, Py)>) -> Py {
    Py::Dict(items.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

fn opt_float(x: Option<f64>) -> Py {
    x.map_or(Py::None, Py::Float)
}

fn dict_mut<'a>(v: &'a mut Py, k: &str) -> &'a mut Py {
    if v.get(k).is_none() {
        v.set(k, Py::dict());
    }
    match v {
        Py::Dict(items) => {
            let i = items.iter().position(|(key, _)| key == k).unwrap_or(0);
            &mut items[i].1
        }
        other => other,
    }
}

/// What a summary prints, and its `--json` record where it has one.
pub struct Summary {
    pub text: String,
    pub json: Option<Py>,
}

impl Summary {
    /// The `--json OUT` file, as `json.dump(out, fh, indent=1)` writes it.
    pub fn json_text(&self) -> Option<String> {
        self.json.as_ref().map(|j| dumps_indent(j, 1))
    }
}

macro_rules! say {
    ($out:expr, $($arg:tt)*) => {{
        $out.push_str(&format!($($arg)*));
        $out.push('\n');
    }};
}

// ------------------------------------------------------------------ h3-graphs

/// One `[INFO] h3/timings/<name> {…}` line, a capture and the rest of a cell.
struct GraphCell {
    rows: Vec<Py>,
    caps: Vec<(String, String, f64)>,
}

/// `GRAPHS.search(line)`.
fn graphs_line(line: &str) -> bool {
    const P: &str = "h3 graphs: ";
    line.match_indices(P).any(|(i, _)| {
        let r = &line[i + P.len()..];
        let parse = || -> Option<()> {
            let d = digits_at(r)?;
            let r = r[d.len()..].strip_prefix(" steps: ")?;
            let d = digits_at(r)?;
            let r = r[d.len()..].strip_prefix(" replayed, ")?;
            let d = digits_at(r)?;
            let r = r[d.len()..].strip_prefix(" captured, ")?;
            let d = digits_at(r)?;
            r[d.len()..].starts_with(" eager").then_some(())
        };
        parse().is_some()
    })
}

/// `CAP.search(line)`: the step, what was captured and the pool high.
fn capture_line(line: &str) -> Option<(String, String, String)> {
    const P: &str = "h3 graphs: step ";
    const POOL: &str = ", graph pool high ";
    for (i, _) in line.match_indices(P) {
        let r = &line[i + P.len()..];
        let Some(step) = digits_at(r) else { continue };
        let Some(r) = r[step.len()..].strip_prefix(" captured: ") else {
            continue;
        };
        // `(.*)` greedy: the last `, graph pool high <[\d.]+> GiB`.
        let mut best = None;
        for (k, _) in r.match_indices(POOL) {
            let v = &r[k + POOL.len()..];
            let n = v
                .bytes()
                .take_while(|c| c.is_ascii_digit() || *c == b'.')
                .count();
            if n > 0 && v[n..].starts_with(" GiB") {
                best = Some((k, v[..n].to_owned()));
            }
        }
        if let Some((k, pool)) = best {
            let step = step.trim_start_matches('0');
            let step = if step.is_empty() { "0" } else { step };
            return Some((step.to_owned(), r[..k].to_owned(), pool));
        }
    }
    None
}

fn graph_cell(d: &Path) -> R<Option<GraphCell>> {
    let mut err = d.join("stderr.log");
    if !err.exists() {
        err = d.join("events.log");
    }
    if !err.exists() {
        return Ok(None);
    }
    let text = py_text(&err)?;
    let mut cell = GraphCell {
        rows: Vec::new(),
        caps: Vec::new(),
    };
    for raw in py_lines(&text) {
        let line = raw.strip_suffix('\n').unwrap_or(raw);
        if graphs_line(line) {
            continue;
        }
        if let Some((step, what, pool)) = capture_line(line) {
            cell.caps.push((step, what, py_float(&pool)?));
            continue;
        }
        if let Some(j) = line.strip_prefix("[INFO] h3/gpu_trace ").and_then(braced) {
            pyfmt::loads(j)?;
            continue;
        }
        if line
            .strip_prefix("[INFO] h3/cold_generation ")
            .and_then(braced)
            .is_some()
        {
            continue;
        }
        if let Some(r) = line.strip_prefix("[INFO] h3/timings/") {
            let n = r.find(py_space).unwrap_or(r.len());
            if n == 0 {
                continue;
            }
            let (name, rest) = r.split_at(n);
            if let Some(j) = rest.strip_prefix(' ').and_then(braced) {
                let mut v = pyfmt::loads(j)?;
                if !matches!(v, Py::Dict(_)) {
                    return Err(format!(
                        "{}: h3/timings/{name} is not an object",
                        err.display()
                    ));
                }
                v.set("name", Py::Str(name.to_owned()));
                cell.rows.push(v);
            }
        }
    }
    Ok(Some(cell))
}

/// `h3-graphs-summary.py <dir>`.
pub fn h3_graphs(root: &Path) -> R<Summary> {
    let mut out = String::new();
    for base in ["cells", "trace"] {
        let p = root.join(base);
        if !p.is_dir() {
            continue;
        }
        for cell in listdir(&p).unwrap_or_default() {
            let Some(c) = graph_cell(&p.join(&cell))? else {
                continue;
            };
            say!(out, "== {base}/{cell}");
            for v in &c.rows {
                let steps: Vec<f64> = get(v, "step_s")?
                    .or(Py::List(Vec::new()))
                    .as_list()
                    .iter()
                    .map(num)
                    .collect::<R<_>>()?;
                let tr = or_dict(get(v, "gpu_trace")?);
                let mut extra = String::new();
                if tr.truthy() {
                    let zero = Py::int(0);
                    let busy = num_f(get_or(&tr, "busy_s", &zero)?, 3)?;
                    let idle = num(get_or(&tr, "idle_s", &zero)?)? * 1e3;
                    extra = format!(
                        " trace: busy {busy} idle {}ms kernels {} host_launches {} memcpy {}",
                        fixed(idle, 1),
                        py_str(get(&tr, "kernel_count")?),
                        py_str(get(&tr, "host_launches")?),
                        dumps(get(&tr, "memcpy")?)
                    );
                }
                let name = index(v, "name")?;
                let med = median(&steps).unwrap_or(0.0);
                say!(
                    out,
                    "  {} total {} denoise {} median step {} peak {} MiB{extra}",
                    ljust(&py_str(name), 28),
                    rjust(&num_f(index(v, "total_s")?, 2)?, 7),
                    rjust(&num_f(index(v, "denoise_s")?, 3)?, 7),
                    fixed(med, 3),
                    py_str(get(v, "peak_mib")?)
                );
            }
            for (step, what, pool) in c.caps.iter().take(2) {
                say!(
                    out,
                    "  capture step {step}: {what} pool {} GiB",
                    fixed(*pool, 2)
                );
            }
            if !c.caps.is_empty() {
                let max = c.caps.iter().map(|c| c.2).fold(f64::NEG_INFINITY, f64::max);
                say!(
                    out,
                    "  max graph pool high {} GiB over {} captures",
                    fixed(max, 2),
                    c.caps.len()
                );
            }
        }
    }
    let h = root.join("hashes.tsv");
    if h.exists() {
        let bytes = std::fs::read(&h).map_err(|e| format!("{}: {e}", h.display()))?;
        let text = String::from_utf8(bytes)
            .map_err(|e| format!("{}: {e}", h.display()))?
            .replace("\r\n", "\n")
            .replace('\r', "\n");
        type Group = Vec<(String, String, String, String)>;
        let mut groups: Vec<((String, String), Group)> = Vec::new();
        for raw in py_lines(&text) {
            let line = raw.strip_suffix('\n').unwrap_or(raw);
            let f: Vec<&str> = line.split('\t').collect();
            let [cell, clip, n, digest] = f[..] else {
                return Err(format!(
                    "{}: not 4 tab-separated fields: {}",
                    h.display(),
                    str_repr(line)
                ));
            };
            let Some((ck, arm)) = cell.split_once('-') else {
                return Err(format!(
                    "{}: cell {} has no `-`",
                    h.display(),
                    str_repr(cell)
                ));
            };
            if clip == "cold" {
                continue;
            }
            let stem = ["-a", "-b", "-c"]
                .iter()
                .find_map(|s| clip.strip_suffix(s))
                .unwrap_or(clip);
            let key = (ck.to_owned(), stem.to_owned());
            let val = (
                arm.to_owned(),
                clip.to_owned(),
                n.to_owned(),
                digest.to_owned(),
            );
            match groups.iter_mut().find(|(k, _)| *k == key) {
                Some((_, v)) => v.push(val),
                None => groups.push((key, vec![val])),
            }
        }
        groups.sort_by(|a, b| a.0.cmp(&b.0));
        say!(
            out,
            "== byte identity (all arms and requests of a checkpoint / prompt / canvas)"
        );
        for ((ck, stem), vals) in &groups {
            let mut digests: Vec<&String> = vals.iter().map(|v| &v.3).collect();
            digests.sort();
            digests.dedup();
            let mut files: Vec<&String> = vals.iter().map(|v| &v.2).collect();
            files.sort();
            files.dedup();
            let files = Py::List(files.into_iter().map(|s| Py::Str(s.clone())).collect());
            let mut line = format!(
                "  {}: {} clips, {} distinct, files {}",
                tuple_repr(&[Py::Str(ck.clone()), Py::Str(stem.clone())]),
                vals.len(),
                digests.len(),
                repr(&files)
            );
            if digests.len() != 1 {
                let all: Vec<String> = vals
                    .iter()
                    .map(|(a, c, n, d)| {
                        tuple_repr(&[
                            Py::Str(a.clone()),
                            Py::Str(c.clone()),
                            Py::Str(n.clone()),
                            Py::Str(d.clone()),
                        ])
                    })
                    .collect();
                line.push_str(&format!("  MISMATCH [{}]", all.join(", ")));
            }
            say!(out, "{line}");
        }
    }
    Ok(Summary {
        text: out,
        json: None,
    })
}

// ------------------------------------------------------------------ tier options

/// `med(xs)`: the median of the values that are not `None`.
fn med(xs: Vec<Py>) -> Option<Py> {
    let xs: Vec<Py> = xs.into_iter().filter(|x| !x.is_none()).collect();
    median_py(&xs)
}

fn opt(v: Option<Py>) -> Py {
    v.unwrap_or(Py::None)
}

/// `span(vals, d)`: `median [min..max]`.
fn span(vals: &[&Py], d: usize) -> R<String> {
    let v: Vec<f64> = vals
        .iter()
        .filter(|x| !x.is_none())
        .map(|x| num(x))
        .collect::<R<_>>()?;
    let Some(m) = median(&v) else {
        return Ok("-".into());
    };
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(format!(
        "{} [{}..{}]",
        fixed(m, d),
        fixed(min, d),
        fixed(max, d)
    ))
}

/// The largest of Python numbers (`max([...])`; the first of equals).
fn py_max<'a>(xs: &[&'a Py]) -> R<&'a Py> {
    let mut best: Option<&Py> = None;
    for x in xs {
        let v = num(x)?;
        if best.is_none_or(|b| v > b.num().unwrap_or(f64::NEG_INFINITY)) {
            best = Some(x);
        }
    }
    best.ok_or_else(|| "max() arg is an empty sequence".into())
}

fn py_min<'a>(xs: &[&'a Py]) -> R<&'a Py> {
    let mut best: Option<&Py> = None;
    for x in xs {
        let v = num(x)?;
        if best.is_none_or(|b| v < b.num().unwrap_or(f64::INFINITY)) {
            best = Some(x);
        }
    }
    best.ok_or_else(|| "min() arg is an empty sequence".into())
}

/// The largest second column of an nvidia-smi CSV (`max(vals)`), if any.
fn nvsmi_max(path: &Path) -> R<Option<f64>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = py_text(path)?;
    let mut best: Option<f64> = None;
    for line in py_lines(&text) {
        let parts: Vec<&str> = line.split(',').map(|x| x.trim_matches(py_space)).collect();
        if parts.len() >= 2 {
            if let Ok(v) = py_float(parts[1]) {
                best = Some(best.map_or(v, |b| b.max(v)));
            }
        }
    }
    Ok(best)
}

/// `h3-tier-options-summary.py <dir> [--json OUT]`.
pub fn h3_tier_options(root: &Path) -> R<Summary> {
    let mut out = String::new();
    let mut js = dict_of(vec![
        ("cells", Py::dict()),
        ("pairs", Py::dict()),
        ("gates", Py::dict()),
        ("check", Py::dict()),
    ]);
    let cells = root.join("cells");
    say!(out, "## Cells (warm; medians over the five prompts)\n");
    say!(
        out,
        "| cell | denoise s | s / forward | VAE decode s | video decode s | end to end s (text cached) | peak GiB (in-process) | peak GiB (nvidia-smi) |"
    );
    say!(out, "|---|---:|---:|---:|---:|---:|---:|---:|");
    for cell in (if cells.is_dir() {
        listdir(&cells)
    } else {
        None
    })
    .unwrap_or_default()
    {
        let p = cells.join(&cell).join("benchmark.json");
        if !p.exists() {
            continue;
        }
        let b = pyfmt::load(&p)?;
        let prompts = or_dict(get(&b, "prompts")?);
        let mut rows: Vec<&Py> = prompts.items().iter().map(|(_, v)| v).collect();
        if rows.is_empty() {
            rows = vec![&b];
        }
        let stage =
            |r: &Py, k: &str| -> R<Py> { Ok(get(&or_dict(get(r, "stage_seconds")?), k)?.clone()) };
        let vae = med(rows
            .iter()
            .map(|r| stage(r, "video_vae"))
            .collect::<R<_>>()?);
        let vdec = med(rows
            .iter()
            .map(|r| stage(r, "video_decode"))
            .collect::<R<_>>()?);
        let den = med(rows
            .iter()
            .map(|r| get(r, "denoise_s").cloned())
            .collect::<R<_>>()?);
        let mut tots = Vec::new();
        for r in &rows {
            let t = get(r, "total_s")?;
            if !t.is_none() {
                let text = get(r, "text_s")?.or(Py::int(0));
                tots.push(t.sub(&text).ok_or("total_s - text_s: not numbers")?);
            }
        }
        let tot = med(tots);
        let mut steps = Vec::new();
        for r in &rows {
            steps.extend(
                get(r, "step_seconds")?
                    .or(Py::List(Vec::new()))
                    .as_list()
                    .to_vec(),
            );
        }
        let fwd = med(steps);
        let peaks: Vec<Py> = rows
            .iter()
            .map(|r| Ok(get(r, "max_device_memory_used_mib")?.or(Py::int(0))))
            .collect::<R<_>>()?;
        let peak_refs: Vec<&Py> = peaks.iter().collect();
        let peak_in = py_max(&peak_refs)?
            .div(&Py::int(1024))
            .ok_or("peak / 1024")?;
        let peak_in = peak_in.or(Py::None);
        let smi = nvsmi_max(&cells.join(&cell).join("nvsmi.csv"))?.map(|v| v / 1024.0);
        let mut per_prompt = Py::dict();
        for (n, r) in prompts.items() {
            per_prompt.set(
                n,
                dict_of(vec![
                    ("denoise_s", get(r, "denoise_s")?.clone()),
                    ("total_s", get(r, "total_s")?.clone()),
                    ("vae_s", stage(r, "video_vae")?),
                    ("peak_mib", get(r, "max_device_memory_used_mib")?.clone()),
                ]),
            );
        }
        let (den, fwd, vae, vdec, tot) = (opt(den), opt(fwd), opt(vae), opt(vdec), opt(tot));
        dict_mut(&mut js, "cells").set(
            &cell,
            dict_of(vec![
                ("denoise_s", den.clone()),
                ("forward_s", fwd.clone()),
                ("vae_s", vae.clone()),
                ("video_decode_s", vdec.clone()),
                ("total_s", tot.clone()),
                ("peak_gib", peak_in.clone()),
                ("nvsmi_peak_gib", opt_float(smi)),
                ("per_prompt", per_prompt),
            ]),
        );
        say!(
            out,
            "| {cell} | {} | {} | {} | {} | {} | {} | {} |",
            f_or_dash(&den, 2)?,
            f_or_dash(&fwd, 3)?,
            f_or_dash(&vae, 2)?,
            f_or_dash(&vdec, 2)?,
            f_or_dash(&tot, 2)?,
            f_or_dash(&peak_in, 1)?,
            f_or_dash(&opt_float(smi), 1)?
        );
    }

    let cdir = root.join("compare").join("clips");
    say!(
        out,
        "\n## Quality per pair (baseline -> candidate; median over prompts, [min..max])\n"
    );
    say!(
        out,
        "| pair | prompts | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |"
    );
    say!(out, "|---|---:|---|---|---|---|");
    let mut pairs = Py::dict();
    for f in (if cdir.is_dir() { listdir(&cdir) } else { None }).unwrap_or_default() {
        let Some((a, b, prompt)) = match_compare_clips(&f, 3) else {
            continue;
        };
        let doc = pyfmt::load(&cdir.join(&f))?;
        let empty = Py::dict();
        let c = get_or(&doc, "context", &empty)?;
        let px = get_or(c, "pixel_metrics", &empty)?;
        let row = dict_of(vec![
            ("psnr", get(px, "psnr_mean")?.clone()),
            ("lpips", get(c, "lpips_mean")?.clone()),
            ("sharpness", get(px, "sharpness_ratio_mean")?.clone()),
            ("jitter", get(px, "temporal_jitter_ratio_mean")?.clone()),
        ]);
        dict_mut(&mut pairs, &format!("{a} -> {b}")).set(&prompt, row);
    }
    for (pair, per) in pairs.items() {
        let col = |k: &str| -> Vec<&Py> { per.items().iter().map(|(_, r)| r.at(k)).collect() };
        say!(
            out,
            "| {pair} | {} | {} | {} | {} | {} |",
            per.items().len(),
            span(&col("psnr"), 2)?,
            span(&col("lpips"), 3)?,
            span(&col("sharpness"), 3)?,
            span(&col("jitter"), 3)?
        );
    }
    js.set("pairs", pairs);

    say!(
        out,
        "\n## Gate (scripts/gpu/gate-policy.toml, kind lossy)\n"
    );
    say!(
        out,
        "| baseline -> candidate | verdict | denoise speed-up | total speed-up | quality | failed hard checks |"
    );
    say!(out, "|---|---|---:|---:|---|---|");
    let gdir = root.join("compare").join("gate");
    for f in (if gdir.is_dir() { listdir(&gdir) } else { None }).unwrap_or_default() {
        let Some((a, b)) = match_gate(&f, 3) else {
            continue;
        };
        let doc = pyfmt::load(&gdir.join(&f))?;
        let empty = Py::dict();
        let v = get_or(get_or(&doc, "context", &empty)?, "verdict", &empty)?;
        let gates = get_or(v, "gates", &empty)?;
        let perf = get_or(get_or(gates, "performance", &empty)?, "detail", &empty)?;
        let q = get_or(gates, "quantitative_quality", &empty)?;
        let no_checks = Py::List(Vec::new());
        let checks = get_or(get_or(q, "detail", &empty)?, "checks", &no_checks)?;
        let mut fails = Vec::new();
        for c in checks.as_list() {
            if get(c, "hard")?.truthy() && !get(c, "ok")?.truthy() {
                let blank = Py::Str(String::new());
                let report = get_or(c, "report", &blank)?
                    .as_str()
                    .ok_or("report is not a string")?;
                let last = report.rsplit("--").next().unwrap_or(report);
                fails.push(format!(
                    "{} {} ({})",
                    py_str(index(c, "check")?),
                    head_chars(&py_str(get_or(c, "value", &blank)?), 6),
                    last.replace(".json", "")
                ));
            }
        }
        dict_mut(&mut js, "gates").set(
            &format!("{a} -> {b}"),
            dict_of(vec![
                ("verdict", get(v, "verdict")?.clone()),
                ("speedup", get(perf, "speedup")?.clone()),
                ("secondary", get(perf, "secondary_speedup")?.clone()),
                ("quality", get(q, "status")?.clone()),
                (
                    "fails",
                    Py::List(fails.iter().map(|s| Py::Str(s.clone())).collect()),
                ),
            ]),
        );
        let joined = fails.join("; ");
        say!(
            out,
            "| {a} -> {b} | {} | {} | {} | {} | {} |",
            py_str(get(v, "verdict")?),
            f_or_dash(get(perf, "speedup")?, 3)?,
            f_or_dash(get(perf, "secondary_speedup")?, 3)?,
            py_str(get(q, "status")?),
            if joined.is_empty() { "-" } else { &joined }
        );
    }

    say!(
        out,
        "\n## NVFP4 VAE check (FASTVIDEO_H3_VAE_CHECK, first tile batch, h3-base bf16 decoder as reference)\n"
    );
    for act in ["static", "dynamic"] {
        let p = root.join("check").join(act).join("vae-lines.txt");
        if !p.exists() {
            continue;
        }
        let text = py_text(&p)?;
        let lines: Vec<String> = py_lines(&text)
            .into_iter()
            .filter(|l| l.contains("check") || l.contains("NVFP4"))
            .map(|l| l.trim_matches(py_space).to_owned())
            .collect();
        dict_mut(&mut js, "check").set(
            act,
            Py::List(lines.iter().map(|s| Py::Str(s.clone())).collect()),
        );
        say!(out, "- **{act}**:");
        for l in &lines {
            say!(out, "  - `{}`", tail_chars(l, 900));
        }
    }
    Ok(Summary {
        text: out,
        json: Some(js),
    })
}

// ------------------------------------------------------------------ trim nvfp4

/// The last `dit_nonlinear <[0-9.]+>` of a text (`re.findall(...)[-1]`).
fn last_dit_nonlinear(text: &str) -> Option<&str> {
    const P: &str = "dit_nonlinear ";
    let mut last = None;
    let mut from = 0;
    while let Some(k) = text[from..].find(P) {
        let s = from + k + P.len();
        let n = text[s..]
            .bytes()
            .take_while(|c| c.is_ascii_digit() || *c == b'.')
            .count();
        if n > 0 {
            last = Some(&text[s..s + n]);
            from = s + n;
        } else {
            from = from + k + 1;
        }
    }
    last
}

/// `h3-trim-nvfp4-summary.py <dir> [--json OUT]`.
pub fn h3_trim_nvfp4(root: &Path) -> R<Summary> {
    let runs = root.join("runs");
    let mut out = String::new();
    let mut js = dict_of(vec![
        ("cells", Py::dict()),
        ("pairs", Py::dict()),
        ("gates", Py::dict()),
        ("parity", Py::dict()),
    ]);
    say!(out, "## Cells (warm; medians over the prompt set)\n");
    say!(
        out,
        "| cell | denoise s | s / forward | total s | VAE decode s | peak MiB (nvidia-smi) | DiT resident GiB |"
    );
    say!(out, "|---|---:|---:|---:|---:|---:|---:|");
    for res in ["480", "768"] {
        for arm in ["bf16", "mxfp8", "nvfp4"] {
            let cell = format!("{arm}-{res}");
            let p = runs.join(&cell).join("benchmark.json");
            if !p.exists() {
                continue;
            }
            let b = pyfmt::load(&p)?;
            let mut steps: Vec<Py> = get(&b, "step_seconds")?
                .or(Py::List(Vec::new()))
                .as_list()
                .to_vec();
            steps.sort_by(|x, y| {
                x.num()
                    .unwrap_or(f64::NAN)
                    .partial_cmp(&y.num().unwrap_or(f64::NAN))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let fwd = if steps.is_empty() {
                Py::Float(f64::NAN)
            } else {
                steps[steps.len() / 2].clone()
            };
            let log = runs.join(&cell).join("stderr.log");
            let dit = if log.exists() {
                last_dit_nonlinear(&py_text(&log)?)
                    .map(py_float)
                    .transpose()?
            } else {
                None
            };
            let prompts = or_dict(get(&b, "prompts")?);
            let peaks: Vec<Py> = prompts
                .items()
                .iter()
                .map(|(_, v)| {
                    Ok(get(v, "max_device_memory_used_mib")?.or(get(v, "peak_memory_mb")?.clone()))
                })
                .collect::<R<Vec<Py>>>()?
                .into_iter()
                .filter(Py::truthy)
                .collect();
            let refs: Vec<&Py> = peaks.iter().collect();
            let peak = if peaks.is_empty() {
                get(&b, "max_device_memory_used_mib")?.clone()
            } else {
                py_min(&refs)?.clone()
            };
            let peak_max = if peaks.is_empty() {
                Py::None
            } else {
                py_max(&refs)?.clone()
            };
            let (den, tot, dec) = (
                index(&b, "denoise_s")?,
                index(&b, "total_s")?,
                index(&b, "decode_s")?,
            );
            dict_mut(&mut js, "cells").set(
                &cell,
                dict_of(vec![
                    ("denoise_s", den.clone()),
                    ("forward_s", fwd.clone()),
                    ("total_s", tot.clone()),
                    ("decode_s", dec.clone()),
                    ("peak_mib", peak.clone()),
                    ("peak_mib_max", peak_max),
                    ("dit_gib", opt_float(dit)),
                ]),
            );
            say!(
                out,
                "| {cell} | {} | {} | {} | {} | {} | {} |",
                num_f(den, 2)?,
                num_f(&fwd, 2)?,
                num_f(tot, 2)?,
                num_f(dec, 2)?,
                num_f(&peak, 0)?,
                dit.map_or("?".to_owned(), pyfmt::float_repr)
            );
        }
    }

    say!(out, "\n## Quality per prompt (baseline -> candidate)\n");
    say!(
        out,
        "| pair | prompt | PSNR dB | LPIPS mean | sharpness ratio | jitter ratio |"
    );
    say!(out, "|---|---|---:|---:|---:|---:|");
    let cdir = runs.join("compare");
    let names = listdir(&cdir).ok_or_else(|| format!("{}: no such directory", cdir.display()))?;
    for f in names {
        let Some((a, b, prompt)) = match_compare_clips(&f, 0) else {
            continue;
        };
        let doc = pyfmt::load(&cdir.join(&f))?;
        let empty = Py::dict();
        let c = get_or(&doc, "context", &empty)?;
        let px = get_or(c, "pixel_metrics", &empty)?;
        let row = dict_of(vec![
            ("psnr", get(px, "psnr_mean")?.clone()),
            ("lpips", get(c, "lpips_mean")?.clone()),
            ("sharpness", get(px, "sharpness_ratio_mean")?.clone()),
            ("jitter", get(px, "temporal_jitter_ratio_mean")?.clone()),
        ]);
        let pair = format!("{a} -> {b}");
        let line = format!(
            "| {pair} | {prompt} | {} | {} | {} | {} |",
            num_f(row.at("psnr"), 2)?,
            num_f(row.at("lpips"), 3)?,
            num_f(row.at("sharpness"), 3)?,
            num_f(row.at("jitter"), 3)?
        );
        dict_mut(dict_mut(&mut js, "pairs"), &pair).set(&prompt, row);
        say!(out, "{line}");
    }

    say!(
        out,
        "\n## Gate (scripts/gpu/gate-policy.toml, kind lossy)\n"
    );
    say!(
        out,
        "| baseline -> candidate | verdict | denoise speed-up | failed hard checks |"
    );
    say!(out, "|---|---|---:|---|");
    let gdir = runs.join("gate");
    let names = listdir(&gdir).ok_or_else(|| format!("{}: no such directory", gdir.display()))?;
    for f in names {
        let Some((a, b)) = match_gate(&f, 0) else {
            continue;
        };
        let doc = pyfmt::load(&gdir.join(&f))?;
        let empty = Py::dict();
        let v = get_or(get_or(&doc, "context", &empty)?, "verdict", &empty)?;
        let gates = get_or(v, "gates", &empty)?;
        let perf = get_or(get_or(gates, "performance", &empty)?, "detail", &empty)?;
        let no_checks = Py::List(Vec::new());
        let checks = get_or(
            get_or(
                get_or(gates, "quantitative_quality", &empty)?,
                "detail",
                &empty,
            )?,
            "checks",
            &no_checks,
        )?;
        let mut fails = Vec::new();
        for c in checks.as_list() {
            if get(c, "hard")?.truthy() && !get(c, "ok")?.truthy() {
                let report = index(c, "report")?
                    .as_str()
                    .ok_or("report is not a string")?;
                fails.push(format!(
                    "{} ({})",
                    py_str(index(c, "check")?),
                    report.rsplit("--").next().unwrap_or(report)
                ));
            }
        }
        let nan = Py::Float(f64::NAN);
        let speed = get_or(perf, "speedup", &nan)?;
        dict_mut(&mut js, "gates").set(
            &format!("{a} -> {b}"),
            dict_of(vec![
                ("verdict", get(v, "verdict")?.clone()),
                ("speedup", get(perf, "speedup")?.clone()),
                (
                    "fails",
                    Py::List(fails.iter().map(|s| Py::Str(s.clone())).collect()),
                ),
            ]),
        );
        let joined = fails.join("; ");
        say!(
            out,
            "| {a} -> {b} | {} | {} | {} |",
            py_str(get(v, "verdict")?),
            num_f(speed, 3)?,
            if joined.is_empty() { "-" } else { &joined }
        );
    }

    say!(
        out,
        "\n## Parity (832x480, one prompt, seed 1024, the same cached text; rel-L2 against bf16)\n"
    );
    let names = [
        "video_step01",
        "video_step02",
        "video_step04",
        "video_step08",
        "video_vel_step01",
        "audio_step01",
        "audio_step08",
        "step00_block_0",
        "step00_block_20",
        "step00_block_41",
    ];
    say!(out, "| tensor | {} |", names.join(" | "));
    say!(out, "|---|{}", "---:|".repeat(names.len()));
    for arm in ["mxfp8", "nvfp4"] {
        let p = root.join(format!("compare-dumps-bf16-{arm}.txt"));
        let text = py_text(&p)?;
        let mut vals: Vec<(String, f64)> = Vec::new();
        for line in py_lines(&text) {
            if let Some((name, v, _)) = match_compare_dumps(line) {
                let v = py_float(v)?;
                match vals.iter_mut().find(|(k, _)| k == name) {
                    Some(slot) => slot.1 = v,
                    None => vals.push((name.to_owned(), v)),
                }
            }
        }
        let val = |n: &str| vals.iter().find(|(k, _)| k == n).map(|(_, v)| *v);
        let mut parity = Py::dict();
        for n in names {
            parity.set(n, opt_float(val(n)));
        }
        dict_mut(&mut js, "parity").set(arm, parity);
        let cols: Vec<String> = names
            .iter()
            .map(|n| exp(val(n).unwrap_or(f64::NAN), 2))
            .collect();
        say!(out, "| bf16 -> {arm} | {} |", cols.join(" | "));
    }
    for arm in ["mxfp8", "nvfp4"] {
        let p = root
            .join("parity")
            .join("compare")
            .join(format!("compare-clips-bf16--{arm}.json"));
        if !p.exists() {
            continue;
        }
        let doc = pyfmt::load(&p)?;
        let empty = Py::dict();
        let c = get_or(&doc, "context", &empty)?;
        let px = get_or(c, "pixel_metrics", &empty)?;
        dict_mut(&mut js, "parity").set(
            &format!("{arm}_clip"),
            dict_of(vec![
                ("psnr", get(px, "psnr_mean")?.clone()),
                ("lpips", get(c, "lpips_mean")?.clone()),
            ]),
        );
        say!(
            out,
            "\nbf16 -> {arm} decoded clip: PSNR {} dB, LPIPS {}",
            num_f(get(px, "psnr_mean")?, 2)?,
            num_f(get(c, "lpips_mean")?, 3)?
        );
    }
    Ok(Summary {
        text: out,
        json: Some(js),
    })
}

// ------------------------------------------------------------------ fasth3 nvfp4

/// `stage(stages, *keys)`: the sum of the truthy stage times whose lowered
/// name holds one of `keys`, else `None`.
fn stage_sum(stages: &Py, keys: &[&str]) -> R<Option<Py>> {
    let mut tot = Py::Float(0.0);
    let mut hit = false;
    for (k, v) in stages.items() {
        let lk = k.to_lowercase();
        if v.truthy() && keys.iter().any(|s| lk.contains(s)) {
            tot = tot
                .add(v)
                .ok_or_else(|| format!("stage {k}: not a number"))?;
            hit = true;
        }
    }
    Ok(hit.then_some(tot))
}

/// The `[denoising_stage|…] Execution completed in <ms> ms` times, in seconds.
fn logged_denoise(text: &str) -> R<Vec<f64>> {
    const P: &str = "[denoising_stage|";
    const MID: &str = "] Execution completed in ";
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(k) = text[from..].find(P) {
        let s = from + k + P.len();
        let close = text[s..].find(']').map(|c| s + c);
        let hit = close.and_then(|c| {
            let r = text[c..].strip_prefix(MID)?;
            let n = r
                .bytes()
                .take_while(|b| b.is_ascii_digit() || *b == b'.')
                .count();
            (n > 0 && r[n..].starts_with(" ms")).then(|| (c + MID.len() + n + 3, r[..n].to_owned()))
        });
        match hit {
            Some((end, ms)) => {
                out.push(py_float(&ms)? / 1000.0);
                from = end;
            }
            None => from = from + k + 1,
        }
    }
    Ok(out)
}

fn nvsmi_peak(root: &Path, t0: f64, t1: f64) -> R<Option<f64>> {
    let p = root.join("nvsmi.csv");
    if !p.exists() {
        return Ok(None);
    }
    let text = py_text(&p)?;
    let mut peak: Option<f64> = None;
    for line in py_lines(&text) {
        let parts: Vec<&str> = line.split(',').map(|x| x.trim_matches(py_space)).collect();
        if parts.len() < 2 {
            continue;
        }
        let (Ok(t), Ok(mib)) = (py_float(parts[0]), py_float(parts[1])) else {
            continue;
        };
        if t0 - 1.0 <= t && t <= t1 + 1.0 {
            peak = Some(peak.map_or(mib, |p| p.max(mib)));
        }
    }
    Ok(peak)
}

/// `re.fullmatch(r"(video|audio)_step\d\d|text_hidden|text_refined|step00_block_(0|1|12|24|36|49)|video_vel_step01", k)`.
fn parity_key(k: &str) -> bool {
    let step = ["video_step", "audio_step"]
        .iter()
        .find_map(|p| k.strip_prefix(p))
        .is_some_and(|d| d.len() == 2 && d.bytes().all(|c| c.is_ascii_digit()));
    step || matches!(
        k,
        "text_hidden"
            | "text_refined"
            | "step00_block_0"
            | "step00_block_1"
            | "step00_block_12"
            | "step00_block_24"
            | "step00_block_36"
            | "step00_block_49"
            | "video_vel_step01"
    )
}

/// `re.search(r"\b(h3_[0-9a-z_]+)\b.*?calls[= ](\d+).*?(?:seconds|s)[= ]([0-9.]+)", line)`.
pub fn phase_line(line: &str) -> Option<(String, String, String)> {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    for (ci, &(i, _)) in chars.iter().enumerate() {
        if ci > 0 && py_word(chars[ci - 1].1) {
            continue;
        }
        let Some(r) = line[i..].strip_prefix("h3_") else {
            continue;
        };
        let n = r
            .bytes()
            .take_while(|c| c.is_ascii_digit() || c.is_ascii_lowercase() || *c == b'_')
            .count();
        if n == 0 || r[n..].chars().next().is_some_and(py_word) {
            continue;
        }
        let name = format!("h3_{}", &r[..n]);
        let rest = &r[n..];
        let calls = rest.match_indices("calls").find_map(|(k, _)| {
            let after = &rest[k + 5..];
            let sep = after.chars().next()?;
            if sep != '=' && sep != ' ' {
                return None;
            }
            let d = digits_at(&after[1..])?;
            Some((k + 5 + 1 + d.len(), d.to_owned()))
        });
        let Some((end, calls)) = calls else { continue };
        let tail = &rest[end..];
        let secs = (0..tail.len())
            .filter(|j| tail.is_char_boundary(*j))
            .find_map(|j| {
                ["seconds", "s"].iter().find_map(|w| {
                    let after = tail[j..].strip_prefix(w)?;
                    let sep = after.chars().next()?;
                    if sep != '=' && sep != ' ' {
                        return None;
                    }
                    let v = &after[1..];
                    let m = v
                        .bytes()
                        .take_while(|c| c.is_ascii_digit() || *c == b'.')
                        .count();
                    (m > 0).then(|| v[..m].to_owned())
                })
            });
        if let Some(s) = secs {
            return Some((name, calls, s));
        }
    }
    None
}

/// `fasth3-nvfp4-summary.py <dir> [--json OUT]`.
pub fn fasth3_nvfp4(root: &Path) -> R<Summary> {
    const SHARP: (f64, f64) = (0.95, 1.08);
    const JITTER: (f64, f64) = (0.85, 1.20);
    let mut out = String::new();
    let mut js = dict_of(vec![
        ("upstream", Py::dict()),
        ("ours", Py::dict()),
        ("parity", Py::dict()),
        ("profile", Py::dict()),
    ]);
    say!(
        out,
        "## Upstream FastVideo (d5253fd, the RTX PRO 6000 page's recipe)\n"
    );
    say!(
        out,
        "| run | ok | wall s | denoise s | s / forward | decode s | device peak MiB |"
    );
    say!(out, "|---|---|---:|---:|---:|---:|---:|");
    for (ck, forwards) in [("v2", 8), ("4s", 4)] {
        let p = root.join(format!("up-{ck}")).join("result.json");
        if !p.exists() {
            continue;
        }
        let r = pyfmt::load(&p)?;
        let mut runs_out = Vec::new();
        let lp = root.join(format!("up-{ck}.log"));
        let logged = if lp.exists() {
            logged_denoise(&py_text(&lp)?)?
        } else {
            Vec::new()
        };
        let no_runs = Py::List(Vec::new());
        for (i, run) in get_or(&r, "runs", &no_runs)?.as_list().iter().enumerate() {
            let st = or_dict(get(run, "stages_s")?);
            let steps8 = matches!(get(run, "steps")?, Py::Int(_) | Py::Float(_))
                && get(run, "steps")?.num() == Some(8.0);
            let mut den = stage_sum(&st, &["denois"])?;
            if den.is_none() && i < logged.len() && !steps8 {
                den = Some(Py::Float(logged[i]));
            }
            let dec = stage_sum(&st, &["decod"])?;
            let zero = Py::int(0);
            let peak = nvsmi_peak(
                root,
                num(get_or(run, "t_start", &zero)?)?,
                num(get_or(run, "t_end", &zero)?)?,
            )?;
            let den_py = opt(den.clone());
            let fwd = match &den {
                Some(d) if d.truthy() => d.div(&Py::int(forwards)).ok_or("denoise / forwards")?,
                _ => Py::None,
            };
            let row = dict_of(vec![
                ("tag", index(run, "tag")?.clone()),
                ("ok", get(run, "ok")?.clone()),
                ("wall_s", get(run, "wall_s")?.clone()),
                ("denoise_s", den_py.clone()),
                ("forward_s", fwd.clone()),
                ("decode_s", opt(dec.clone())),
                ("peak_mib", opt_float(peak)),
                ("error", get(run, "error")?.clone()),
                ("stages", st.clone()),
            ]);
            runs_out.push(row.clone());
            let err = get(run, "error")?.or(Py::Str(String::new()));
            let err = err.as_str().ok_or("error is not a string")?.to_owned();
            let why = if steps8 {
                "refused (grid points)".to_owned()
            } else if den_py.truthy() {
                "decode failed (upstream VAE compile)".to_owned()
            } else {
                first_line(&err).unwrap_or("").to_owned()
            };
            let ok = if get(run, "ok")?.truthy() {
                "yes".to_owned()
            } else {
                format!("no: {}", head_chars(&why, 90))
            };
            say!(
                out,
                "| {} | {ok} | {} | {} | {} | {} | {} |",
                py_str(index(run, "tag")?),
                f_or_dash(row.at("wall_s"), 2)?,
                f_or_dash(&den_py, 2)?,
                f_or_dash(&fwd, 2)?,
                f_or_dash(&opt(dec), 2)?,
                f_or_dash(&opt_float(peak), 0)?
            );
        }
        dict_mut(&mut js, "upstream").set(
            ck,
            dict_of(vec![
                ("load_s", get(&r, "load_s")?.clone()),
                ("runs", Py::List(runs_out)),
            ]),
        );
        let zero = Py::int(0);
        say!(
            out,
            "\nload {} s ({ck})\n",
            num_f(get_or(&r, "load_s", &zero)?, 1)?
        );
    }

    say!(
        out,
        "\n## Ours (fv-gpucheck h3 gen, one process per arm, warm after the 480p h3-demo generation)\n"
    );
    say!(
        out,
        "| checkpoint | arm | prompt | denoise s | s / forward (median) | decode s | total s | device peak MiB |"
    );
    say!(out, "|---|---|---|---:|---:|---:|---:|---:|");
    for ck in ["v2", "4s"] {
        for arm in ["nvfp4", "nvfp4ffn-mxfp8", "dense-mxfp8"] {
            let p = root
                .join(format!("ours-{ck}"))
                .join(arm)
                .join("benchmark.json");
            if !p.exists() {
                continue;
            }
            let b = pyfmt::load(&p)?;
            let mut rows = Py::dict();
            for (name, v) in or_dict(get(&b, "prompts")?).items() {
                let steps: Vec<Py> = get(v, "step_seconds")?
                    .or(Py::List(Vec::new()))
                    .as_list()
                    .to_vec();
                let fwd = opt(median_py(&steps));
                let row = dict_of(vec![
                    ("denoise_s", get(v, "denoise_s")?.clone()),
                    ("forward_s", fwd),
                    ("decode_s", get(v, "decode_s")?.clone()),
                    ("total_s", get(v, "total_s")?.clone()),
                    (
                        "peak_mib",
                        get(v, "max_device_memory_used_mib")?.or(get(v, "peak_memory_mb")?.clone()),
                    ),
                ]);
                say!(
                    out,
                    "| {ck} | {arm} | {name} | {} | {} | {} | {} | {} |",
                    num_f(row.at("denoise_s"), 2)?,
                    num_f(row.at("forward_s"), 3)?,
                    num_f(row.at("decode_s"), 2)?,
                    num_f(row.at("total_s"), 2)?,
                    num_f(&row.at("peak_mib").or(Py::int(0)), 0)?
                );
                rows.set(name, row);
            }
            dict_mut(dict_mut(&mut js, "ours"), ck).set(
                arm,
                dict_of(vec![
                    ("prompts", rows),
                    ("warmup", get(&b, "warmup_requests")?.clone()),
                    ("load_s", get(&b, "load_s")?.clone()),
                ]),
            );
        }
    }

    say!(
        out,
        "\n## Parity: ours (NVFP4 attention + FFN, VSA) on the reference's noise and text, 832x480x124\n"
    );
    for ck in ["v2", "4s"] {
        let p = root.join(format!("compare-dumps-{ck}.txt"));
        if !p.exists() {
            continue;
        }
        let text = py_text(&p)?;
        let mut rel = Py::dict();
        for line in py_lines(&text) {
            let Some((name, v, rest)) = match_compare_dumps(line) else {
                continue;
            };
            // `\s+cosine (\S+)` after the value.
            let ws = rest.len() - rest.trim_start_matches(py_space).len();
            let ok = ws > 0
                && rest[ws..]
                    .strip_prefix("cosine ")
                    .is_some_and(|c| c.chars().next().is_some_and(|ch| !py_space(ch)));
            if ok {
                rel.set(name, Py::Float(py_float(v)?));
            }
        }
        let mut keys: Vec<&String> = rel
            .items()
            .iter()
            .map(|(k, _)| k)
            .filter(|k| parity_key(k))
            .collect();
        keys.sort();
        let parts: Vec<String> = keys
            .iter()
            .map(|k| Ok(format!("{k} {}", exp(num(rel.at(k))?, 2))))
            .collect::<R<_>>()?;
        say!(out, "**{ck}**: {}\n", parts.join(", "));
        let mut entry = dict_of(vec![("rel_l2", rel.clone())]);
        let a = root.join(format!("av-psnr-{ck}.json"));
        if a.exists() {
            let av = pyfmt::load(&a)?;
            entry.set("av_psnr", av.clone());
            say!(out, "clip PSNR: {}\n", head_chars(&dumps(&av), 400));
        }
        let cdir = root.join(format!("ours-{ck}")).join("compare");
        if cdir.is_dir() {
            for f in listdir(&cdir).unwrap_or_default() {
                if !f.ends_with(".json") {
                    continue;
                }
                let doc = pyfmt::load(&cdir.join(&f))?;
                let empty = Py::dict();
                let c = get_or(&doc, "context", &empty)?;
                let px = get_or(c, "pixel_metrics", &empty)?;
                let (sh, ji) = (
                    get(px, "sharpness_ratio_mean")?,
                    get(px, "temporal_jitter_ratio_mean")?,
                );
                let within = |v: &Py, (lo, hi): (f64, f64)| -> R<bool> {
                    let x = num(v)?;
                    Ok(lo <= x && x <= hi)
                };
                let ok =
                    !sh.is_none() && !ji.is_none() && within(sh, SHARP)? && within(ji, JITTER)?;
                entry.set(
                    "clip",
                    dict_of(vec![
                        ("psnr", get(px, "psnr_mean")?.clone()),
                        ("lpips", get(c, "lpips_mean")?.clone()),
                        ("sharpness", sh.clone()),
                        ("jitter", ji.clone()),
                        (
                            "gate_hard_quality",
                            Py::Str(if ok { "pass" } else { "fail" }.into()),
                        ),
                    ]),
                );
                say!(
                    out,
                    "clip vs upstream: PSNR {}, LPIPS {}, sharpness {}, jitter {} -> lossy gate hard quality checks: {}\n",
                    py_str(get(px, "psnr_mean")?),
                    py_str(get(c, "lpips_mean")?),
                    py_str(sh),
                    py_str(ji),
                    if ok { "pass" } else { "FAIL" }
                );
            }
        }
        dict_mut(&mut js, "parity").set(ck, entry);
    }

    say!(
        out,
        "\n## Per-block profile (ours, V2, 480p, FASTVIDEO_PROFILE=1 phases: synchronizing, so slower than timed)\n"
    );
    for ck in ["v2", "4s"] {
        let log = root
            .join(format!("ours-{ck}"))
            .join("profile")
            .join("phases.log");
        if !log.exists() {
            continue;
        }
        let text = py_text(&log)?;
        let mut phases: Vec<(String, u64, f64)> = Vec::new();
        for line in py_lines(&text) {
            if let Some((k, c, s)) = phase_line(line) {
                let c: u64 = c.parse().map_err(|_| format!("calls {c}"))?;
                let s = py_float(&s)?;
                match phases.iter_mut().find(|p| p.0 == k) {
                    Some(slot) => {
                        slot.1 = c;
                        slot.2 = s;
                    }
                    None => phases.push((k, c, s)),
                }
            }
        }
        let mut pj = Py::dict();
        for (k, c, s) in &phases {
            pj.set(k, Py::List(vec![Py::int(*c as i64), Py::Float(*s)]));
        }
        dict_mut(&mut js, "profile").set(ck, pj);
        if !phases.is_empty() {
            say!(out, "| phase | calls | total s | ms per call |");
            say!(out, "|---|---:|---:|---:|");
            let mut sorted = phases.clone();
            sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            for (k, c, s) in sorted {
                say!(
                    out,
                    "| {k} | {c} | {} | {} |",
                    fixed(s, 2),
                    fixed(1000.0 * s / (c.max(1) as f64), 2)
                );
            }
        }
    }
    Ok(Summary {
        text: out,
        json: Some(js),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_file_name_patterns_match_as_pythons_re() {
        assert_eq!(
            match_compare_clips(
                "compare-clips-v2-bf16vae-480--v2-nvvae-480-talking-head.json",
                3
            ),
            Some((
                "v2-bf16vae-480".into(),
                "v2-nvvae-480".into(),
                "talking-head".into()
            ))
        );
        assert_eq!(
            match_compare_clips(
                "compare-clips-bf16-480--mxfp8-480-ltx-newsbroadcast.json",
                0
            ),
            Some((
                "bf16-480".into(),
                "mxfp8-480".into(),
                "ltx-newsbroadcast".into()
            ))
        );
        // `.+?` is lazy: the first `-ddd--` ends the first group, the first
        // `-ddd-` the second (Python: ('a-480', 'b-480', '-c-480-p')).
        assert_eq!(
            match_compare_clips("compare-clips-a-480--b-480--c-480-p.json", 3),
            Some(("a-480".into(), "b-480".into(), "-c-480-p".into()))
        );
        assert_eq!(
            match_compare_clips("compare-clips-a--b-480-p.json", 3),
            None
        );
        assert_eq!(
            match_gate("gate-bf16-480--nvfp4-480.json", 0),
            Some(("bf16-480".into(), "nvfp4-480".into()))
        );
        assert_eq!(
            match_gate("gate-trim-bf16-768--trim-nv-768.json", 3),
            Some(("trim-bf16-768".into(), "trim-nv-768".into()))
        );
        assert_eq!(match_gate("gate-a-48--b-480.json", 3), None);
    }

    #[test]
    fn graph_lines_match_as_pythons_re() {
        assert!(graphs_line(
            "x h3 graphs: 8 steps: 7 replayed, 1 captured, 0 eager"
        ));
        assert!(!graphs_line("h3 graphs: 8 steps: 7 replayed, 1 captured"));
        assert_eq!(
            capture_line("h3 graphs: step 03 captured: a, b, graph pool high 1.5 GiB, graph pool high 2.25 GiB tail"),
            Some(("3".into(), "a, b, graph pool high 1.5 GiB".into(), "2.25".into()))
        );
    }

    #[test]
    fn the_phase_profile_line_matches_as_pythons_re() {
        assert_eq!(
            phase_line("[phase] h3_attn_fwd calls=96 seconds=1.25\n"),
            Some(("h3_attn_fwd".into(), "96".into(), "1.25".into()))
        );
        // `(?:seconds|s)` takes the first `s=`: as the script does.
        assert_eq!(
            phase_line("h3_ffn calls 12 bytes=10 seconds=3"),
            Some(("h3_ffn".into(), "12".into(), "10".into()))
        );
        assert_eq!(phase_line("xh3_ffn calls=1 s=1"), None);
        assert_eq!(phase_line("h3_ffnA calls=1 s=1"), None);
    }

    #[test]
    fn compare_dumps_and_parity_keys() {
        assert_eq!(
            match_compare_dumps("compare-dumps video_step01   rel_l2 1.2e-03  cosine 0.99\n"),
            Some(("video_step01", "1.2e-03", "  cosine 0.99\n"))
        );
        assert!(parity_key("video_step04"));
        assert!(!parity_key("video_step4"));
        assert!(parity_key("step00_block_12"));
        assert!(!parity_key("step00_block_2"));
        assert_eq!(
            last_dit_nonlinear("dit_nonlinear 1.5 x dit_nonlinear 2.25"),
            Some("2.25")
        );
        assert_eq!(
            logged_denoise("[denoising_stage|x] Execution completed in 1500.5 ms\n").unwrap(),
            vec![1.5005]
        );
    }
}
