# Synthetic Pi and native Claude gateway qualification

On 2026-09-15 the **actual** Pi 0.84.4 and native Claude Code 2.1.272 binaries
completed a two-request conversation through the gateway compiler, run capability
and budget service, and Anthropic HTTP/SSE adapter. Each executed its real Bash
tool, ran Python to write `synthetic-result.txt` containing `42`, returned a
`tool_result`, and printed `SYNTHETIC_CLI_COMPLETE` with exit status zero.

The upstream was HTTPX `MockTransport`, not Anthropic. No billable calls occurred.
All upstream requests used the fixed model `claude-sonnet-4-6` and the trusted
server setting `output_config.effort=high`. This checks configuration preservation
and protocol execution, not model reasoning quality.

## Isolation and inputs

The binaries ran as the unprivileged worker in the clean Debian outer VM, in a
new network namespace with **only loopback**. Each invocation had a new private
HOME, configuration directory, working directory and explicitly constructed
environment. Claude used `--bare`, disabled settings/MCP/skills/session persistence,
and only Bash. Pi disabled extensions, skills, context files, themes, templates,
and session persistence, and only enabled Bash. The harness records the exact
arguments. No host home, existing session, provider key, mailbox, repository or
project configuration was copied.

`cli-runtime-inputs.json` records the binary/tree hashes and public Python
dependency versions. Node and Pi came from the existing stopped
`gmail-search-pi:local` image, copied without mounting volumes or running that
container. Only `/usr/local/bin/node` and the installed public Pi package tree
were extracted. The native Claude binary was copied from the known versioned
installation without running it against the user's home. Version commands ran
in the clean worker. Native Claude is x86_64 ELF; Node is v24.20.0.

The archive hash describes the recorded transfer, not a vendor signature. The Pi
tree hash iterates sorted relative paths, hashing each path, a NUL separator,
`F` plus its SHA256 digest (or `L` plus the symlink target), then another NUL.
Public runtime inputs remain outside the repository; do not commit the binaries.

## Request compatibility

Unchanged strict compilation rejected all four captured requests. The new
**server-selected**, versioned profiles accept only the observed compatibility
hints before passing the result to the original closed compiler:

- Pi `pi-0.84.4`: validate and remove ephemeral cache hints. No provider cache
  directive reaches upstream; nonzero cache usage remains refused.
- Claude `claude-2.1.272`: additionally remove bounded `metadata.user_id`, validate
  and discard the CLI effort hint, and remove the exact known JSON Schema 2020-12
  dialect URI from the tool schema. The URI is never fetched. Only the trusted
  immutable run profile selects the effort sent upstream.

The default profile is still `strict`. A request cannot choose its profile,
model, effort, owner or upstream. Remote images, hosted tools, unknown schema
URIs and unlisted fields are still rejected. Client beta/query/header hints are
not forwarded by the HTTP adapter.

The [official effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort)
lists low, medium, high and max for Sonnet 4.6. The closed server capability table
supports those values for that exact model; xhigh and unqualified model/effort
pairs are rejected. Unit tests cover all four values; the actual CLI spike used
**high only**.

## Reproduction

Prepare the clean outer worker following `README.md`. Stage only the recorded
public inputs at `/home/worker/cli-spike/bin` and `lib/pi-coding-agent`. Put public
HTTPX dependencies under `cli-spike/python` and the seven credential-free gateway
modules (`__init__`, `inference`, `cli_compat`, `provider`, `provider_http`,
`capabilities`, `registry`) under `python/gateway`. Copy the pure `inference.py`
module to the `cli-spike` root for the unchanged-strict comparison. Stage this
standalone harness there as well.

Inside the worker, run:

```sh
sudo unshare --net -- /bin/sh -c 'ip link set lo up; exec runuser -u worker -- env -i PATH=/usr/bin:/bin HOME=/home/worker/cli-spike/probe-home python3 /home/worker/cli-spike/cli-compatibility-spike.py'
```

The harness prints its private results directory. It records synthetic request
bodies, field/header names, upstream accepted fields, and the two CLI outcomes.
`cli-spike-summary.json` and `cli-spike-observations.json` preserve the successful
high-effort run. Tests use text-sanitized request shapes under
`tests/fixtures/cli_compat`, retaining the real structural compatibility fields.

For inner-VM work, start each fixed-profile synthetic gateway as the unprivileged
worker, outside the extra network namespace, binding **127.0.0.1 only**:

```sh
env -i PATH=/usr/bin:/bin HOME=/home/worker/cli-spike/probe-home python3 /home/worker/cli-spike/cli-compatibility-spike.py --serve-runtime pi --port 18081 --ready-file /home/worker/cli-spike/gateway-pi.ready
env -i PATH=/usr/bin:/bin HOME=/home/worker/cli-spike/probe-home python3 /home/worker/cli-spike/cli-compatibility-spike.py --serve-runtime claude --port 18082 --ready-file /home/worker/cli-spike/gateway-claude.ready
```

Each server stops after 600 seconds, preserves observations on SIGTERM, and
publishes endpoint, PID, profile and output directory in its ready file. The
synthetic HTTP harness does not authenticate external callers: it binds one
fixed internal run and uses no real provider. It must never be exposed publicly.
The production request boundary and the guest vsock relay are separate work.

## Remaining qualification

The recorded run covers one Bash/Python tool cycle per CLI, not complete Pi or
Claude feature parity. It does not qualify provider accounts, live model
behavior, tool portfolios, sessions/workspace restoration, real-mail retrieval,
browser login, or guest adversarial isolation. Inner Firecracker execution and
vsock transport are subsequent tests; these results establish outer-worker CLI
and synthetic provider protocol compatibility only.
