//! `torch.save` checkpoints (`.pt` / `.pth`) without Python: torch's zip
//! format (torch >= 1.6) and its legacy format, through a small pickle
//! interpreter that understands what `torch.save` writes for (nested) state
//! dicts: dicts / `OrderedDict`s, tensors (`torch._utils._rebuild_tensor_v2`,
//! `_rebuild_parameter`), and plain Python values (ints, floats, strings,
//! lists, tuples, `None`, bools) around them.
//!
//! Tensors are read as float32 (float32 storages bit for bit; bfloat16,
//! float16 and float64 converted). Only contiguous row-major tensors are
//! accepted, which is what `torch.save` writes for weights.
//!
//! Used by the LPIPS port (`alexnet-owt-7be5be79.pth`, `lpips/alex.pth`) and
//! the LongLive-Plug H3 few-step LoRA (`generator_lora.pt`, whose tensors sit
//! under `"student_lora"` beside a training-config dict).

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::LoaderError;

type Result<T> = std::result::Result<T, LoaderError>;

fn err(s: impl Into<String>) -> LoaderError {
    LoaderError::Message(s.into())
}

/// A tensor read from a `.pth`, as float32.
#[derive(Clone, Debug, PartialEq)]
pub struct PthTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
enum P {
    None,
    Bool(bool),
    Int(i64),
    #[allow(dead_code)]
    Float(f64),
    Str(String),
    Bytes,
    Tuple(Vec<P>),
    List(Vec<P>),
    Dict(Vec<(P, P)>),
    Global(String),
    Storage {
        key: String,
        dtype: String,
    },
    Tensor(TensorRef),
    Mark,
    Opaque,
}

#[derive(Clone, Debug)]
struct TensorRef {
    key: String,
    dtype: String,
    offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

fn int(p: &P) -> Result<i64> {
    match p {
        P::Int(i) => Ok(*i),
        P::Bool(b) => Ok(i64::from(*b)),
        other => Err(err(format!("pickle: expected an int, got {other:?}"))),
    }
}

fn usizes(p: &P) -> Result<Vec<usize>> {
    match p {
        P::Tuple(v) | P::List(v) => v.iter().map(|x| Ok(int(x)? as usize)).collect(),
        other => Err(err(format!(
            "pickle: expected a tuple of ints, got {other:?}"
        ))),
    }
}

fn reduce(f: P, args: P) -> Result<P> {
    let P::Global(name) = &f else {
        return Ok(P::Opaque);
    };
    let args = match args {
        P::Tuple(a) => a,
        other => return Err(err(format!("pickle: REDUCE args {other:?}"))),
    };
    Ok(match name.as_str() {
        "collections OrderedDict" | "builtins dict" => match args.first() {
            None => P::Dict(Vec::new()),
            Some(P::List(items)) => P::Dict(
                items
                    .iter()
                    .filter_map(|kv| match kv {
                        P::List(p) | P::Tuple(p) if p.len() == 2 => {
                            Some((p[0].clone(), p[1].clone()))
                        }
                        _ => None,
                    })
                    .collect(),
            ),
            Some(_) => P::Opaque,
        },
        "torch._utils _rebuild_tensor_v2" | "torch._utils _rebuild_tensor" => {
            let (key, dtype) = match args.first() {
                Some(P::Storage { key, dtype }) => (key.clone(), dtype.clone()),
                other => return Err(err(format!("pickle: tensor storage {other:?}"))),
            };
            P::Tensor(TensorRef {
                key,
                dtype,
                offset: int(args.get(1).unwrap_or(&P::Int(0)))? as usize,
                shape: usizes(args.get(2).unwrap_or(&P::Tuple(Vec::new())))?,
                stride: usizes(args.get(3).unwrap_or(&P::Tuple(Vec::new())))?,
            })
        }
        "torch._utils _rebuild_parameter" => args.into_iter().next().unwrap_or(P::None),
        // `Tensor.__reduce_ex__` of a tensor that carries Python state:
        // `_rebuild_from_type_v2(func, type, args, state)` = `func(*args)`
        // re-typed (to `torch.Tensor` here); the state is not needed.
        "torch._tensor _rebuild_from_type_v2" => {
            let mut it = args.into_iter();
            match (it.next(), it.next(), it.next()) {
                (Some(func @ P::Global(_)), Some(_), Some(inner @ P::Tuple(_))) => {
                    reduce(func, inner)?
                }
                other => return Err(err(format!("pickle: _rebuild_from_type_v2 args {other:?}"))),
            }
        }
        _ => P::Opaque,
    })
}

/// Run one pickle from `r` (stops at STOP). Enough of protocols 2-5 for
/// `torch.save` of state dicts and the plain config values beside them.
fn unpickle(r: &mut impl Read) -> Result<P> {
    let mut stack: Vec<P> = Vec::new();
    let mut memo: BTreeMap<u32, P> = BTreeMap::new();
    let mut b1 = [0u8; 1];
    let byte = |r: &mut dyn Read| -> Result<u8> {
        let mut b = [0u8; 1];
        r.read_exact(&mut b)?;
        Ok(b[0])
    };
    let bytes = |r: &mut dyn Read, n: usize| -> Result<Vec<u8>> {
        let mut v = vec![0u8; n];
        r.read_exact(&mut v)?;
        Ok(v)
    };
    let u32le = |r: &mut dyn Read| -> Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    };
    let u64le = |r: &mut dyn Read| -> Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    };
    let line = |r: &mut dyn Read| -> Result<String> {
        let mut s = Vec::new();
        loop {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)?;
            if b[0] == b'\n' {
                break;
            }
            s.push(b[0]);
        }
        Ok(String::from_utf8_lossy(&s).into_owned())
    };
    let pop_mark = |stack: &mut Vec<P>| -> Result<Vec<P>> {
        let pos = stack
            .iter()
            .rposition(|p| matches!(p, P::Mark))
            .ok_or_else(|| err("pickle: no MARK"))?;
        let items = stack.split_off(pos + 1);
        stack.pop();
        Ok(items)
    };
    let pop = |stack: &mut Vec<P>| stack.pop().ok_or_else(|| err("pickle: stack underflow"));
    let le_int = |b: &[u8]| -> i64 {
        let n = b.len();
        let mut v: i128 = 0;
        for (i, x) in b.iter().enumerate() {
            v |= i128::from(*x) << (8 * i);
        }
        if n > 0 && n < 16 && b[n - 1] & 0x80 != 0 {
            v -= 1i128 << (8 * n);
        }
        v as i64
    };
    loop {
        r.read_exact(&mut b1)?;
        let op = b1[0];
        match op {
            // PROTO, FRAME
            0x80 => {
                byte(r)?;
            }
            0x95 => {
                bytes(r, 8)?;
            }
            b'.' => return pop(&mut stack),
            // GLOBAL, STACK_GLOBAL
            b'c' => {
                let m = line(r)?;
                let n = line(r)?;
                stack.push(P::Global(format!("{m} {n}")));
            }
            0x93 => {
                let n = pop(&mut stack)?;
                let m = pop(&mut stack)?;
                match (m, n) {
                    (P::Str(m), P::Str(n)) => stack.push(P::Global(format!("{m} {n}"))),
                    _ => stack.push(P::Opaque),
                }
            }
            // BINPUT, LONG_BINPUT, MEMOIZE, PUT
            b'q' => {
                let i = u32::from(byte(r)?);
                memo.insert(i, stack.last().cloned().unwrap_or(P::None));
            }
            b'r' => {
                let i = u32le(r)?;
                memo.insert(i, stack.last().cloned().unwrap_or(P::None));
            }
            0x94 => {
                let i = memo.len() as u32;
                memo.insert(i, stack.last().cloned().unwrap_or(P::None));
            }
            b'p' => {
                let i: u32 = line(r)?
                    .trim()
                    .parse()
                    .map_err(|_| err("pickle: PUT index"))?;
                memo.insert(i, stack.last().cloned().unwrap_or(P::None));
            }
            // BINGET, LONG_BINGET, GET
            b'h' | b'j' | b'g' => {
                let i = match op {
                    b'h' => u32::from(byte(r)?),
                    b'j' => u32le(r)?,
                    _ => line(r)?
                        .trim()
                        .parse()
                        .map_err(|_| err("pickle: GET index"))?,
                };
                stack.push(
                    memo.get(&i)
                        .cloned()
                        .ok_or_else(|| err(format!("pickle: memo {i}")))?,
                );
            }
            b')' => stack.push(P::Tuple(Vec::new())),
            b']' => stack.push(P::List(Vec::new())),
            b'}' => stack.push(P::Dict(Vec::new())),
            0x8f => stack.push(P::List(Vec::new())), // EMPTY_SET (as a list)
            b'(' => stack.push(P::Mark),
            b'N' => stack.push(P::None),
            0x88 => stack.push(P::Bool(true)),
            0x89 => stack.push(P::Bool(false)),
            b'K' => stack.push(P::Int(i64::from(byte(r)?))),
            b'M' => {
                let b = bytes(r, 2)?;
                stack.push(P::Int(i64::from(u16::from_le_bytes([b[0], b[1]]))));
            }
            b'J' => {
                let b = bytes(r, 4)?;
                stack.push(P::Int(i64::from(i32::from_le_bytes([
                    b[0], b[1], b[2], b[3],
                ]))));
            }
            // LONG1, LONG4
            0x8a => {
                let n = byte(r)? as usize;
                let b = bytes(r, n)?;
                stack.push(P::Int(le_int(&b)));
            }
            0x8b => {
                let n = u32le(r)? as usize;
                let b = bytes(r, n)?;
                stack.push(P::Int(le_int(&b)));
            }
            // INT / LONG (text): "I01\n" is True, "I00\n" False.
            b'I' | b'L' => {
                let s = line(r)?;
                let s = s.trim().trim_end_matches('L');
                stack.push(match s {
                    "01" => P::Bool(true),
                    "00" => P::Bool(false),
                    _ => P::Int(s.parse().map_err(|_| err(format!("pickle: INT {s}")))?),
                });
            }
            b'G' => {
                let b = bytes(r, 8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(&b);
                stack.push(P::Float(f64::from_be_bytes(a)));
            }
            b'F' => {
                let s = line(r)?;
                stack.push(P::Float(
                    s.trim()
                        .parse()
                        .map_err(|_| err(format!("pickle: FLOAT {s}")))?,
                ));
            }
            // BINUNICODE, BINSTRING, SHORT_BINUNICODE, SHORT_BINSTRING, BINUNICODE8
            b'X' | b'T' => {
                let n = u32le(r)? as usize;
                stack.push(P::Str(String::from_utf8_lossy(&bytes(r, n)?).into_owned()));
            }
            0x8c | b'U' => {
                let n = byte(r)? as usize;
                stack.push(P::Str(String::from_utf8_lossy(&bytes(r, n)?).into_owned()));
            }
            0x8d => {
                let n = u64le(r)? as usize;
                stack.push(P::Str(String::from_utf8_lossy(&bytes(r, n)?).into_owned()));
            }
            b'V' => {
                let s = line(r)?;
                stack.push(P::Str(s));
            }
            // SHORT_BINBYTES, BINBYTES, BINBYTES8, BYTEARRAY8
            b'C' => {
                let n = byte(r)? as usize;
                bytes(r, n)?;
                stack.push(P::Bytes);
            }
            b'B' => {
                let n = u32le(r)? as usize;
                bytes(r, n)?;
                stack.push(P::Bytes);
            }
            0x8e | 0x96 => {
                let n = u64le(r)? as usize;
                bytes(r, n)?;
                stack.push(P::Bytes);
            }
            b't' => {
                let items = pop_mark(&mut stack)?;
                stack.push(P::Tuple(items));
            }
            0x85 => {
                let a = pop(&mut stack)?;
                stack.push(P::Tuple(vec![a]));
            }
            0x86 => {
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(P::Tuple(vec![a, b]));
            }
            0x87 => {
                let c = pop(&mut stack)?;
                let b = pop(&mut stack)?;
                let a = pop(&mut stack)?;
                stack.push(P::Tuple(vec![a, b, c]));
            }
            // POP, POP_MARK, DUP
            b'0' => {
                pop(&mut stack)?;
            }
            b'1' => {
                pop_mark(&mut stack)?;
            }
            b'2' => {
                let top = stack
                    .last()
                    .cloned()
                    .ok_or_else(|| err("pickle: DUP on empty"))?;
                stack.push(top);
            }
            // BINPERSID
            b'Q' => {
                let pid = pop(&mut stack)?;
                let P::Tuple(t) = pid else {
                    return Err(err(format!("pickle: persistent id {pid:?}")));
                };
                let key = match t.get(2) {
                    Some(P::Str(s)) => s.clone(),
                    other => return Err(err(format!("pickle: storage key {other:?}"))),
                };
                let dtype = match t.get(1) {
                    Some(P::Global(g)) => g.clone(),
                    other => return Err(err(format!("pickle: storage type {other:?}"))),
                };
                stack.push(P::Storage { key, dtype });
            }
            b'R' => {
                let args = pop(&mut stack)?;
                let f = pop(&mut stack)?;
                stack.push(reduce(f, args)?);
            }
            // NEWOBJ: cls.__new__(cls, *args): nothing we read is built this way.
            0x81 => {
                pop(&mut stack)?;
                pop(&mut stack)?;
                stack.push(P::Opaque);
            }
            b'b' => {
                pop(&mut stack)?; // BUILD state (`_metadata`): not needed
            }
            b'a' => {
                let v = pop(&mut stack)?;
                if let Some(P::List(l)) = stack.last_mut() {
                    l.push(v);
                }
            }
            // APPENDS, ADDITEMS
            b'e' | 0x90 => {
                let items = pop_mark(&mut stack)?;
                if let Some(P::List(l)) = stack.last_mut() {
                    l.extend(items);
                }
            }
            // FROZENSET
            0x91 => {
                let items = pop_mark(&mut stack)?;
                stack.push(P::List(items));
            }
            b's' => {
                let v = pop(&mut stack)?;
                let k = pop(&mut stack)?;
                if let Some(P::Dict(d)) = stack.last_mut() {
                    d.push((k, v));
                }
            }
            b'u' => {
                let items = pop_mark(&mut stack)?;
                if let Some(P::Dict(d)) = stack.last_mut() {
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        d.push((k, v));
                    }
                }
            }
            other => return Err(err(format!("pickle: unsupported opcode 0x{other:02x}"))),
        }
    }
}

/// The tensors of `root`, or of the first `nested` key (in order) whose value
/// is a dict of tensors when `root` itself holds none.
fn state_dict(root: P, nested: &[&str]) -> Result<Vec<(String, TensorRef)>> {
    let P::Dict(items) = root else {
        return Err(err("pth: the root object is not a dict"));
    };
    let tensors = |items: &[(P, P)]| -> Vec<(String, TensorRef)> {
        items
            .iter()
            .filter_map(|(k, v)| match (k, v) {
                (P::Str(k), P::Tensor(t)) => Some((k.clone(), t.clone())),
                _ => None,
            })
            .collect()
    };
    let top = tensors(&items);
    if !top.is_empty() {
        return Ok(top);
    }
    for want in nested {
        for (k, v) in &items {
            if let (P::Str(k), P::Dict(inner)) = (k, v) {
                if k == want {
                    return Ok(tensors(inner));
                }
            }
        }
    }
    let keys: Vec<String> = items
        .iter()
        .filter_map(|(k, _)| match k {
            P::Str(s) => Some(s.clone()),
            _ => None,
        })
        .collect();
    Err(err(format!(
        "pth: no tensors at the root or under {nested:?} (root keys {keys:?})"
    )))
}

/// Element width and decoder of a storage type (`torch FloatStorage`, ...).
fn storage_width(dtype: &str) -> Result<usize> {
    let name = dtype.rsplit(' ').next().unwrap_or(dtype);
    Ok(match name {
        "FloatStorage" => 4,
        "BFloat16Storage" | "HalfStorage" => 2,
        "DoubleStorage" => 8,
        other => return Err(err(format!("pth: {other} is not a float storage"))),
    })
}

fn materialize(t: &TensorRef, storage: &[u8]) -> Result<PthTensor> {
    let width = storage_width(&t.dtype)?;
    let numel: usize = t.shape.iter().product();
    // Contiguous row-major only (what torch.save writes for these weights).
    let mut want = vec![1usize; t.shape.len()];
    for i in (0..t.shape.len().saturating_sub(1)).rev() {
        want[i] = want[i + 1] * t.shape[i + 1];
    }
    if numel > 1 && t.stride != want {
        return Err(err(format!(
            "pth: non-contiguous stride {:?} for shape {:?}",
            t.stride, t.shape
        )));
    }
    let start = t.offset * width;
    let end = start + numel * width;
    if end > storage.len() {
        return Err(err(format!(
            "pth: tensor past the end of storage {}",
            t.key
        )));
    }
    let raw = &storage[start..end];
    let name = t.dtype.rsplit(' ').next().unwrap_or(&t.dtype);
    let data = match name {
        "FloatStorage" => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        "BFloat16Storage" => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes(*c)) << 16))
            .collect(),
        "HalfStorage" => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
            .collect(),
        _ => raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c) as f32)
            .collect(),
    };
    Ok(PthTensor {
        shape: t.shape.clone(),
        data,
    })
}

/// IEEE binary16 to binary32 (exact).
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let bits = match (exp, mant) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: normalize.
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// Zip entries (name → (offset of data, size)), stored (uncompressed) only.
fn zip_entries(f: &mut std::fs::File) -> Result<BTreeMap<String, (u64, u64)>> {
    let len = f.seek(SeekFrom::End(0))?;
    let tail = len.min(65_557);
    f.seek(SeekFrom::Start(len - tail))?;
    let mut buf = vec![0u8; tail as usize];
    f.read_exact(&mut buf)?;
    let eocd = (0..buf.len().saturating_sub(21))
        .rev()
        .find(|&i| buf[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| err("zip: no end of central directory"))?;
    let rd16 = |b: &[u8], i: usize| u64::from(u16::from_le_bytes([b[i], b[i + 1]]));
    let rd32 =
        |b: &[u8], i: usize| u64::from(u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]));
    let rd64 = |b: &[u8], i: usize| {
        let mut a = [0u8; 8];
        a.copy_from_slice(&b[i..i + 8]);
        u64::from_le_bytes(a)
    };
    let mut count = rd16(&buf, eocd + 10);
    let mut cd_size = rd32(&buf, eocd + 12);
    let mut cd_off = rd32(&buf, eocd + 16);
    // ZIP64 (torch writes it for large archives): the locator precedes the EOCD.
    if (cd_off == 0xffff_ffff || count == 0xffff || cd_size == 0xffff_ffff) && eocd >= 20 {
        let loc = eocd - 20;
        if buf[loc..loc + 4] == [0x50, 0x4b, 0x06, 0x07] {
            let rec = rd64(&buf, loc + 8);
            f.seek(SeekFrom::Start(rec))?;
            let mut z = [0u8; 56];
            f.read_exact(&mut z)?;
            count = rd64(&z, 32);
            cd_size = rd64(&z, 40);
            cd_off = rd64(&z, 48);
        }
    }
    f.seek(SeekFrom::Start(cd_off))?;
    let mut cd = vec![0u8; cd_size as usize];
    f.read_exact(&mut cd)?;
    let mut out = BTreeMap::new();
    let mut i = 0usize;
    for _ in 0..count {
        if cd.get(i..i + 4) != Some(&[0x50, 0x4b, 0x01, 0x02][..]) {
            return Err(err("zip: bad central directory entry"));
        }
        let method = rd16(&cd, i + 10);
        let mut csize = rd32(&cd, i + 20);
        let mut usize_ = rd32(&cd, i + 24);
        let nlen = rd16(&cd, i + 28) as usize;
        let xlen = rd16(&cd, i + 30) as usize;
        let clen = rd16(&cd, i + 32) as usize;
        let mut local = rd32(&cd, i + 42);
        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + nlen]).into_owned();
        // ZIP64 extra field: sizes / offset that did not fit.
        let mut x = i + 46 + nlen;
        let xend = x + xlen;
        while x + 4 <= xend {
            let id = rd16(&cd, x);
            let sz = rd16(&cd, x + 2) as usize;
            if id == 1 {
                let mut p = x + 4;
                let mut next = || {
                    let v = rd64(&cd, p);
                    p += 8;
                    v
                };
                if usize_ == 0xffff_ffff {
                    usize_ = next();
                }
                if csize == 0xffff_ffff {
                    csize = next();
                }
                if local == 0xffff_ffff {
                    local = next();
                }
            }
            x += 4 + sz;
        }
        if method != 0 || csize != usize_ {
            return Err(err(format!("zip: {name} is compressed (method {method})")));
        }
        // Local header: fixed 30 bytes + its own name / extra lengths.
        f.seek(SeekFrom::Start(local))?;
        let mut lh = [0u8; 30];
        f.read_exact(&mut lh)?;
        let data = local + 30 + rd16(&lh, 26) + rd16(&lh, 28);
        out.insert(name, (data, csize));
        i += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

/// Every float tensor of a `torch.save`d state dict whose key passes `keep`,
/// from either the zip format (torch >= 1.6) or the legacy one.
pub fn read_pth(path: &Path, keep: impl Fn(&str) -> bool) -> Result<BTreeMap<String, PthTensor>> {
    read_pth_nested(path, &[], keep)
}

/// [`read_pth`] for a checkpoint whose tensors may sit one level down: the
/// root dict's tensors if it has any, else those of the first of `nested`
/// (e.g. `["student_lora", "generator_lora", "state_dict"]`) that is a dict.
pub fn read_pth_nested(
    path: &Path,
    nested: &[&str],
    keep: impl Fn(&str) -> bool,
) -> Result<BTreeMap<String, PthTensor>> {
    let ctx = |e: LoaderError| err(format!("{}: {e}", path.display()));
    let mut f = std::fs::File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    f.seek(SeekFrom::Start(0))?;
    let mut out = BTreeMap::new();
    if magic == [0x50, 0x4b, 0x03, 0x04] {
        let entries = zip_entries(&mut f).map_err(ctx)?;
        let (pkl, _) = entries
            .iter()
            .find(|(n, _)| n.ends_with("/data.pkl") || n.as_str() == "data.pkl")
            .map(|(n, v)| (n.clone(), *v))
            .ok_or_else(|| err(format!("{}: no data.pkl", path.display())))?;
        let prefix = pkl.trim_end_matches("data.pkl").to_string();
        let (off, size) = entries[&pkl];
        f.seek(SeekFrom::Start(off))?;
        let mut raw = vec![0u8; size as usize];
        f.read_exact(&mut raw)?;
        let root = unpickle(&mut raw.as_slice()).map_err(ctx)?;
        for (name, t) in state_dict(root, nested).map_err(ctx)? {
            if !keep(&name) {
                continue;
            }
            let (off, size) = *entries
                .get(&format!("{prefix}data/{}", t.key))
                .ok_or_else(|| err(format!("{}: storage {} missing", path.display(), t.key)))?;
            f.seek(SeekFrom::Start(off))?;
            let mut storage = vec![0u8; size as usize];
            f.read_exact(&mut storage)?;
            out.insert(name, materialize(&t, &storage).map_err(ctx)?);
        }
    } else {
        // Legacy: magic, protocol, sys_info, the object, the storage keys,
        // then each storage as <u64 numel><raw little-endian data>.
        let mut all = Vec::new();
        f.read_to_end(&mut all)?;
        let mut r = all.as_slice();
        for _ in 0..3 {
            unpickle(&mut r).map_err(ctx)?;
        }
        let root = unpickle(&mut r).map_err(ctx)?;
        let tensors = state_dict(root, nested).map_err(ctx)?;
        let keys = match unpickle(&mut r).map_err(ctx)? {
            P::List(k) => k
                .into_iter()
                .filter_map(|k| if let P::Str(s) = k { Some(s) } else { None })
                .collect::<Vec<_>>(),
            other => return Err(err(format!("{}: storage keys {other:?}", path.display()))),
        };
        // Each storage's element width comes from the tensors that use it.
        let widths: BTreeMap<String, usize> = tensors
            .iter()
            .filter_map(|(_, t)| storage_width(&t.dtype).ok().map(|w| (t.key.clone(), w)))
            .collect();
        let mut storages: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        for key in keys {
            let mut n = [0u8; 8];
            r.read_exact(&mut n)?;
            let bytes = u64::from_le_bytes(n) as usize * widths.get(&key).copied().unwrap_or(4);
            if r.len() < bytes {
                return Err(err(format!("{}: storage {key} truncated", path.display())));
            }
            storages.insert(key, r[..bytes].to_vec());
            r = &r[bytes..];
        }
        for (name, t) in tensors {
            if keep(&name) {
                let s = storages
                    .get(&t.key)
                    .ok_or_else(|| err(format!("{}: storage {} missing", path.display(), t.key)))?;
                out.insert(name, materialize(&t, s).map_err(ctx)?);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A stored (uncompressed) zip of `files`, as torch writes them.
    pub(crate) fn zip(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut cd = Vec::new();
        for (name, data) in files {
            let off = out.len() as u32;
            let mut lh = vec![
                0x50, 0x4b, 0x03, 0x04, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            lh.extend((data.len() as u32).to_le_bytes());
            lh.extend((data.len() as u32).to_le_bytes());
            lh.extend((name.len() as u16).to_le_bytes());
            lh.extend(0u16.to_le_bytes());
            out.extend(&lh);
            out.extend(name.as_bytes());
            out.extend(data);
            let mut c = vec![
                0x50, 0x4b, 0x01, 0x02, 20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            c.extend((data.len() as u32).to_le_bytes());
            c.extend((data.len() as u32).to_le_bytes());
            c.extend((name.len() as u16).to_le_bytes());
            c.extend([0u8; 12]);
            c.extend(off.to_le_bytes());
            c.extend(name.as_bytes());
            cd.extend(c);
        }
        let cd_off = out.len() as u32;
        out.extend(&cd);
        let mut e = vec![0x50, 0x4b, 0x05, 0x06, 0, 0, 0, 0];
        e.extend((files.len() as u16).to_le_bytes());
        e.extend((files.len() as u16).to_le_bytes());
        e.extend((cd.len() as u32).to_le_bytes());
        e.extend(cd_off.to_le_bytes());
        e.extend([0u8; 2]);
        out.extend(e);
        out
    }

    fn temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("fv-pth-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.pt");
        std::fs::write(&f, bytes).unwrap();
        f
    }

    /// `_rebuild_tensor_v2((storage, <type>, <key>, cpu, n), offset, shape, stride, False, OrderedDict())`.
    fn tensor_pickle(storage_type: &str, key: &str, offset: u8, shape: (u8, u8)) -> Vec<u8> {
        let mut p = Vec::new();
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n(");
        p.extend(b"(X\x07\x00\x00\x00storage");
        p.extend(format!("ctorch\n{storage_type}\n").as_bytes());
        p.extend(b"X");
        p.extend((key.len() as u32).to_le_bytes());
        p.extend(key.as_bytes());
        p.extend(b"X\x03\x00\x00\x00cpuK\x05tQ");
        p.extend([
            b'K', offset, b'K', shape.0, b'K', shape.1, 0x86, b'K', shape.1, b'K', 1, 0x86,
        ]);
        p.extend(b"\x89ccollections\nOrderedDict\n)RtR");
        p
    }

    #[test]
    fn zip_pth_root_state_dict() {
        // {"w": tensor [2, 2] at storage offset 1}.
        let mut p: Vec<u8> = vec![0x80, 2];
        p.extend(b"ccollections\nOrderedDict\nq\x00)Rq\x01(X\x01\x00\x00\x00wq\x02");
        p.extend(tensor_pickle("FloatStorage", "0", 1, (2, 2)));
        p.extend(b"u.");
        let storage: Vec<u8> = [9.0f32, 1.0, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = temp(
            "root",
            &zip(&[("archive/data.pkl", p), ("archive/data/0", storage)]),
        );
        let t = read_pth(&f, |_| true).unwrap();
        assert_eq!(t["w"].shape, vec![2, 2]);
        assert_eq!(
            t["w"].data,
            vec![1.0, 2.0, 3.0, 4.0],
            "offset 1 skips the first float"
        );
        let _ = std::fs::remove_dir_all(f.parent().unwrap());
    }

    #[test]
    fn nested_lora_dict_beside_a_config() {
        // {"schema_version": 1, "config": {"lr": 1e-5, "targets": ["a"], "x": None},
        //  "student_lora": {"m.lora_A.weight": bf16 [1, 2], "m.lora_B.weight": f32 [2, 1]}}
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'q', 0, b'('];
        p.extend(b"X\x0e\x00\x00\x00schema_versionK\x01");
        p.extend(b"X\x06\x00\x00\x00config}(X\x02\x00\x00\x00lrG");
        p.extend(1e-5f64.to_be_bytes());
        p.extend(b"X\x07\x00\x00\x00targets](X\x01\x00\x00\x00aeX\x01\x00\x00\x00xNu");
        p.extend(b"X\x0c\x00\x00\x00student_lora}(");
        p.extend(b"X\x0f\x00\x00\x00m.lora_A.weight");
        p.extend(tensor_pickle("BFloat16Storage", "0", 0, (1, 2)));
        p.extend(b"X\x0f\x00\x00\x00m.lora_B.weight");
        p.extend(tensor_pickle("FloatStorage", "1", 0, (2, 1)));
        p.extend(b"uu.");
        let a: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let b: Vec<u8> = [0.25f32, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = temp(
            "nested",
            &zip(&[
                ("generator_lora/data.pkl", p),
                ("generator_lora/data/0", a),
                ("generator_lora/data/1", b),
            ]),
        );
        let t = read_pth_nested(&f, &["generator_lora", "student_lora"], |_| true).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t["m.lora_A.weight"].shape, vec![1, 2]);
        assert_eq!(t["m.lora_A.weight"].data, vec![1.5, -2.0]);
        assert_eq!(t["m.lora_B.weight"].shape, vec![2, 1]);
        assert_eq!(t["m.lora_B.weight"].data, vec![0.25, 4.0]);
        // Without the nested key the root has no tensors: a clear error.
        let e = read_pth(&f, |_| true).unwrap_err().to_string();
        assert!(e.contains("no tensors"), "{e}");
        assert!(e.contains("student_lora"), "{e}");
        let _ = std::fs::remove_dir_all(f.parent().unwrap());
    }

    #[test]
    fn tensors_rebuilt_from_type_v2_as_the_h3_release_writes_them() {
        // generator_lora.pt wraps each tensor as
        // _rebuild_from_type_v2(_rebuild_tensor_v2, torch.Tensor, (args...), {}).
        let mut p: Vec<u8> = vec![0x80, 2, b'}', b'('];
        p.extend(b"X\x0c\x00\x00\x00student_lora}(X\x0f\x00\x00\x00m.lora_A.weight");
        p.extend(b"ctorch._tensor\n_rebuild_from_type_v2\n(");
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\nctorch\nTensor\n(");
        p.extend(b"(X\x07\x00\x00\x00storagectorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x02tQ");
        p.extend([b'K', 0, b'K', 1, b'K', 2, 0x86, b'K', 2, b'K', 1, 0x86]);
        p.extend(b"\x89ccollections\nOrderedDict\n)Rt}tRuu.");
        let a: Vec<u8> = [0.5f32, -3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let f = temp(
            "fromtype",
            &zip(&[("generator_lora/data.pkl", p), ("generator_lora/data/0", a)]),
        );
        let t = read_pth_nested(&f, &["student_lora"], |_| true).unwrap();
        assert_eq!(t["m.lora_A.weight"].shape, vec![1, 2]);
        assert_eq!(t["m.lora_A.weight"].data, vec![0.5, -3.0]);
        let _ = std::fs::remove_dir_all(f.parent().unwrap());
    }

    #[test]
    fn half_floats_convert_exactly() {
        // (binary16 bits, value)
        let cases: [(u16, f32); 6] = [
            (0x0000, 0.0),
            (0x3c00, 1.0),
            (0xc100, -2.5),
            (0x7bff, 65504.0),
            (0x0400, 6.103_515_6e-5),
            (0x0001, 5.960_464_5e-8),
        ];
        for (h, v) in cases {
            assert_eq!(f16_to_f32(h), v, "{h:#06x}");
        }
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
