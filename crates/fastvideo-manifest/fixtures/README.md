# Fixtures

The acceptance fixtures of the registry's WO-001, kept here because the
formats live here (`docs/adr/0005`..`0007`). `model-registry-rs` runs the same
files as its acceptance tests of this crate.

| Directory | Contents | Written by |
|---|---|---|
| `names/valid.txt` | names that parse and `Display` back byte-identical | hand |
| `names/invalid.txt` | `name # ErrorKind`, every `NameError` kind at least once | hand |
| `model-toml/valid/*.toml` | `model.toml` files that parse and round-trip | hand |
| `model-toml/invalid/*.toml` + `.expect` | files that fail; `.expect` is `Kind path` (`ModelTomlError::kind`, `::path`) | hand |
| `manifests/*.json` + `.digest` | manifests and their canonical digest (hex); `vendor.canonical.json` is the exact hashed bytes | `gen_fixtures.py` |
| `manifests/invalid/*.json` | manifests `Manifest::from_json` rejects | `gen_fixtures.py` |
| `keys/` | TEST ONLY Ed25519 keys, PKCS#8 / SPKI PEM | `gen_fixtures.py` |
| `lineage/<case>/{vendor,fork1,fork2}.{json,model.toml,digest}` | a signed three-manifest chain (`ok`) and three broken variants | `gen_fixtures.py` |

`gen_fixtures.py` is an independent oracle (Python `blake3`, `cryptography`,
`json`): it shares no code with the crate, so agreement on every digest and
signature is the evidence that the implementation matches the format.
