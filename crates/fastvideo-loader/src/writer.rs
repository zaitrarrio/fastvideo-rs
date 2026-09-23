//! Streaming safetensors writer.
//!
//! Rewriting a checkpoint — dropping the tensors a port never reads, storing
//! float32 weights as the bfloat16 they are loaded as, laying tensors out in
//! the order a streaming loader asks for them — must not need the checkpoint
//! in memory: the inputs are 50 GB. So the layout is declared up front (the
//! header needs every offset before the first byte of data), and tensors are
//! then written one at a time, in that order, each checked against its
//! declared size. The file appears under its final name only when complete.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::{LazyDType, LoaderError};

/// One tensor of the file to be written. `dtype` must have a byte size.
#[derive(Debug, Clone)]
pub struct TensorSpec {
    pub name: String,
    pub dtype: LazyDType,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, dtype: LazyDType, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape,
        }
    }

    fn bytes(&self) -> Result<usize, LoaderError> {
        let size = self.dtype.size().ok_or_else(|| {
            LoaderError::Message(format!(
                "{}: {:?} has no fixed element size",
                self.name, self.dtype
            ))
        })?;
        Ok(self.shape.iter().product::<usize>() * size)
    }
}

pub struct SafetensorsWriter {
    out: BufWriter<File>,
    path: PathBuf,
    part: PathBuf,
    /// `(name, byte length)` in file order.
    layout: Vec<(String, usize)>,
    next: usize,
}

fn io(path: &Path, e: impl std::fmt::Display) -> LoaderError {
    LoaderError::Message(format!("{}: {e}", path.display()))
}

impl SafetensorsWriter {
    /// Write the header for `specs` (file order = slice order) and return a
    /// writer expecting their data in that order. `metadata` lands in
    /// `__metadata__`: say where the file came from and what was done to it.
    pub fn create(
        path: &Path,
        specs: &[TensorSpec],
        metadata: &[(&str, &str)],
    ) -> Result<Self, LoaderError> {
        let mut seen = std::collections::HashSet::new();
        let mut header = String::from("{");
        if !metadata.is_empty() {
            let meta: serde_json::Map<String, serde_json::Value> = metadata
                .iter()
                .map(|(k, v)| {
                    (
                        (*k).to_string(),
                        serde_json::Value::String((*v).to_string()),
                    )
                })
                .collect();
            header.push_str(&format!(
                "\"__metadata__\":{},",
                serde_json::Value::Object(meta)
            ));
        }
        let mut layout = Vec::with_capacity(specs.len());
        let mut offset = 0usize;
        for (i, spec) in specs.iter().enumerate() {
            if spec.name == "__metadata__" || !seen.insert(spec.name.as_str()) {
                return Err(io(
                    path,
                    format!("duplicate or reserved tensor name '{}'", spec.name),
                ));
            }
            let len = spec.bytes()?;
            if i > 0 {
                header.push(',');
            }
            // Built by hand so the header lists tensors in file order (serde's
            // map would sort them); names still go through serde for escaping.
            header.push_str(&format!(
                "{}:{{\"dtype\":\"{}\",\"shape\":{},\"data_offsets\":[{},{}]}}",
                serde_json::Value::String(spec.name.clone()),
                spec.dtype.as_str(),
                serde_json::to_string(&spec.shape).map_err(|e| io(path, e))?,
                offset,
                offset + len
            ));
            layout.push((spec.name.clone(), len));
            offset += len;
        }
        header.push('}');
        // Pad with spaces so the data starts 8-byte aligned, as the format asks.
        while (8 + header.len()) % 8 != 0 {
            header.push(' ');
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| io(path, e))?;
        }
        let mut part = path.as_os_str().to_owned();
        part.push(".part");
        let part = PathBuf::from(part);
        let mut out =
            BufWriter::with_capacity(8 << 20, File::create(&part).map_err(|e| io(&part, e))?);
        out.write_all(&(header.len() as u64).to_le_bytes())
            .map_err(|e| io(&part, e))?;
        out.write_all(header.as_bytes()).map_err(|e| io(&part, e))?;
        Ok(Self {
            out,
            path: path.to_path_buf(),
            part,
            layout,
            next: 0,
        })
    }

    /// The tensor the writer expects next, if any.
    pub fn expecting(&self) -> Option<&str> {
        self.layout.get(self.next).map(|(n, _)| n.as_str())
    }

    /// Append the next tensor's little-endian bytes. Out-of-order names and
    /// wrong lengths are errors: either would silently shift every later tensor.
    pub fn write(&mut self, name: &str, bytes: &[u8]) -> Result<(), LoaderError> {
        let (want, len) = self.layout.get(self.next).ok_or_else(|| {
            io(
                &self.path,
                format!("'{name}' written after the last declared tensor"),
            )
        })?;
        if want != name {
            return Err(io(
                &self.path,
                format!("expected tensor '{want}' next, got '{name}'"),
            ));
        }
        if bytes.len() != *len {
            return Err(io(
                &self.path,
                format!("'{name}': {} bytes written, {len} declared", bytes.len()),
            ));
        }
        self.out.write_all(bytes).map_err(|e| io(&self.part, e))?;
        self.next += 1;
        Ok(())
    }

    /// Flush and move the file to its final name. Fails if tensors are missing.
    pub fn finish(mut self) -> Result<(), LoaderError> {
        if let Some(missing) = self.expecting() {
            return Err(io(
                &self.path,
                format!(
                    "unfinished: '{missing}' and {} more not written",
                    self.layout.len() - self.next - 1
                ),
            ));
        }
        self.out.flush().map_err(|e| io(&self.part, e))?;
        self.out
            .get_ref()
            .sync_all()
            .map_err(|e| io(&self.part, e))?;
        std::fs::rename(&self.part, &self.path).map_err(|e| io(&self.path, e))
    }
}

impl Drop for SafetensorsWriter {
    fn drop(&mut self) {
        // An abandoned writer leaves no file that looks complete.
        if self.next < self.layout.len() {
            let _ = std::fs::remove_file(&self.part);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LazyStore;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("fv-writer-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn what_is_written_reads_back_through_the_lazy_store() {
        let d = tmp("roundtrip");
        let path = d.join("slim.safetensors");
        let a: Vec<u8> = (0..12u8).collect(); // [2, 3] bf16
        let b: Vec<u8> = [1.5f32, -2.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let specs = [
            TensorSpec::new("layers.1.w", LazyDType::BF16, vec![2, 3]),
            TensorSpec::new("layers.0.b \"quoted\"", LazyDType::F32, vec![2]),
            TensorSpec::new("q", LazyDType::F8E4M3, vec![4]),
        ];
        let mut w = SafetensorsWriter::create(&path, &specs, &[("source", "unit test")]).unwrap();
        assert_eq!(w.expecting(), Some("layers.1.w"));
        w.write("layers.1.w", &a).unwrap();
        w.write("layers.0.b \"quoted\"", &b).unwrap();
        w.write("q", &[9, 8, 7, 6]).unwrap();
        assert!(
            !path.exists(),
            "the final name must not appear before finish()"
        );
        w.finish().unwrap();

        let s = LazyStore::open(&d).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.view("layers.1.w").unwrap().bytes, a.as_slice());
        assert_eq!(
            s.to_f32("layers.0.b \"quoted\"").unwrap().1,
            vec![1.5, -2.0]
        );
        assert_eq!(
            (s.view("q").unwrap().dtype, s.view("q").unwrap().bytes),
            (&LazyDType::F8E4M3, &[9u8, 8, 7, 6][..])
        );
        // Data starts 8-byte aligned and the header keeps file order.
        let raw = std::fs::read(&path).unwrap();
        let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + hlen) % 8, 0);
        let header = std::str::from_utf8(&raw[8..8 + hlen]).unwrap();
        assert!(header.find("layers.1.w").unwrap() < header.find("layers.0.b").unwrap());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn order_and_sizes_are_enforced_and_nothing_half_written_survives() {
        let d = tmp("strict");
        let path = d.join("x.safetensors");
        let specs = [
            TensorSpec::new("a", LazyDType::F32, vec![1]),
            TensorSpec::new("b", LazyDType::F32, vec![2]),
        ];
        let mut w = SafetensorsWriter::create(&path, &specs, &[]).unwrap();
        assert!(w
            .write("b", &[0; 8])
            .unwrap_err()
            .to_string()
            .contains("expected tensor 'a'"));
        assert!(w
            .write("a", &[0; 3])
            .unwrap_err()
            .to_string()
            .contains("4 declared"));
        w.write("a", &[0; 4]).unwrap();
        let e = w.finish().unwrap_err().to_string();
        assert!(e.contains("unfinished") && e.contains("'b'"), "{e}");
        assert!(!path.exists());
        assert!(
            std::fs::read_dir(&d).unwrap().next().is_none(),
            "the .part file must be removed"
        );

        let dup = [
            TensorSpec::new("a", LazyDType::F32, vec![1]),
            TensorSpec::new("a", LazyDType::F32, vec![1]),
        ];
        assert!(SafetensorsWriter::create(&path, &dup, &[]).is_err());
        let odd = [TensorSpec::new(
            "p",
            LazyDType::Other("I64".into()),
            vec![1],
        )];
        assert!(SafetensorsWriter::create(&path, &odd, &[]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
