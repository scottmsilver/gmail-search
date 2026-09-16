"""Explicit production entrypoint; never falls back to the legacy public server."""
import asyncio
import os
import sys

from .invited_config import load_runtime_config


async def run(config):
    from .invited_listeners import serve_apps
    from .invited_runtime import open_runtime
    async with open_runtime(config) as runtime:
        await serve_apps(runtime.browser_app,runtime.gateway_app)


def main():
    try:
        if sys.argv[1:] not in ([],['--check-config']):
            raise ValueError('Unsupported invocation')
        config=load_runtime_config(os.environ.get('GMS_INVITED_CONFIG',''))
        if sys.argv[1:] == ['--check-config']:
            print('Private invited configuration is valid; deployment readiness was not checked.')
            return
        asyncio.run(run(config))
    except KeyboardInterrupt:
        return
    except Exception:
        # Configuration, DB and network exceptions can contain private values.
        print('Invited service startup or shutdown failed; public admission is closed.',file=sys.stderr)
        raise SystemExit(1) from None


if __name__=='__main__':
    main()
