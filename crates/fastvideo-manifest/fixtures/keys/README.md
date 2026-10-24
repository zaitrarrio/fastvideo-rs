# Test keys — TEST ONLY

Ed25519 key pairs used by this crate's tests and by the registry's acceptance
tests. They are derived from fixed, public seeds (`0x11 * 32` and
`0x22 * 32`; see `../gen_fixtures.py`) and sign nothing outside the fixtures.
Never use them for a real manifest.

| kid | private key (PKCS#8 PEM) | public key (SPKI PEM) |
|---|---|---|
| `test-key-1` | `test-key-1.pem` | `test-key-1.pub.pem` |
| `test-key-2` | `test-key-2.pem` | `test-key-2.pub.pem` |

`fixtures/manifests/fork.json` and `fixtures/lineage/*/{vendor,fork1}.json`
are signed with `test-key-1`; `fixtures/lineage/*/fork2.json` with
`test-key-2`. The signatures were produced by the Python oracle, not by this
crate, so verifying them is a cross-implementation check.
