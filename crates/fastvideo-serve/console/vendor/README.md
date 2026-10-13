# Vendored console assets

The console is self-hosted (served from the fv-serve binary, `script-src
'self'`), so third-party browser code is vendored here and embedded with
`include_str!` (crates/fastvideo-serve/src/console.rs).

| file | what | version | licence | source |
|---|---|---|---|---|
| `hls.light.min.js` | hls.js light build (HLS over Media Source Extensions; no subtitles, alternate audio, EME or CMCD) for RTMPS/SRT output playback (`outputs.js`) | 1.5.20 | Apache-2.0 (`hls.js-LICENSE.txt`) | npm `hls.js@1.5.20` (`https://registry.npmjs.org/hls.js/-/hls.js-1.5.20.tgz`, sha256 `3791f532701590ec10ecb4a2abf6ea63c65ae9ce91de4cc6f6553ba4629605ce`), file `package/dist/hls.light.min.js`, verbatim |

`hls.light.min.js` sha256: `4a582e5e99d6a424c5f50b5e716d70dc256cd9b6c264cdba95c5ea2a4df9328a`.
It is loaded on first use as a classic script (it defines `window.Hls`)
and run without its Web Worker (`enableWorker: false`; a blob worker
would need `worker-src blob:` in the console's CSP).

To update: download the new npm tarball, check its integrity against the
registry, copy `package/dist/hls.light.min.js` and `package/LICENSE`, and
update this table and the sha256 above.
