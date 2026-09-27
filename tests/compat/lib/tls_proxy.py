#!/usr/bin/env python3
"""TLS front for a plain-HTTP fv-serve (tests/compat/run.sh).

Real clients such as `fal_client` only speak https, and design §7.5 runs
the suites over TLS with a self-signed CA (`SSL_CERT_FILE`,
`NODE_EXTRA_CA_CERTS`). This terminates TLS on 127.0.0.1:<listen> and pipes
bytes to 127.0.0.1:<upstream>. Prints `READY` once listening.

    tls_proxy.py --listen 18443 --upstream 18080 --cert cert.pem --key key.pem
"""

import argparse
import asyncio
import ssl


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except Exception:  # noqa: BLE001 - a closed peer ends the pipe
        pass
    finally:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, required=True)
    ap.add_argument("--upstream", type=int, required=True)
    ap.add_argument("--cert", required=True)
    ap.add_argument("--key", required=True)
    a = ap.parse_args()
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(a.cert, a.key)

    async def handle(cr, cw):
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", a.upstream)
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))

    srv = await asyncio.start_server(handle, "127.0.0.1", a.listen, ssl=ctx)
    print("READY", flush=True)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
