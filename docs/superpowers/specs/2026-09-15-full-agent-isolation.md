# Proposal: full-capability public Gmail agents

Status: approved for implementation on 2026-09-15. Shared-database gateway and
isolated-worker prerequisites are being implemented and tested. Full agents and
invited-user access are not deployed. A separate OAuth landing-navigation fix
has been deployed to the existing owner-only public release.

## Objective

Restore the useful private-agent capabilities to the public app: Pi and Claude
runtimes, model selection, shell/Python/file tools, attachment analysis, charts,
spreadsheets, published artifacts, multi-turn workspaces, workflow subagents,
battles and SQL analysis. **Admission policy confirmed by the user: invited
users, each seeing only their own mail.** Multi-user isolation is a launch
requirement for this replacement, not a deferred feature. The existing live
site remains owner-only until the replacement passes these checks.

Security assumption: a malicious email can persuade the model to run arbitrary
commands. Correctness must not depend on the model refusing that instruction.
Contain that execution with infrastructure and capability checks.

## Invited-user admission and mailbox ownership

- Administrators can invite or revoke users. An invitation names the exact
  verified Google email; possessing an invitation URL alone grants no access.
  A valid broker login must match an active invitation. Bind the account to a
  stable server-owned user ID and verified provider subject. Changing an email
  or reconnecting an account must never reassign someone else's mailbox.
- Sign-in proves identity. Connecting Gmail is a separate consent flow, bound
  by signed, one-use state to that signed-in user and the expected account.
  Verify the returned Google account before storing credentials or ingesting
  mail. New users begin with an empty mailbox, never the bootstrap owner's data.
  Connect/disconnect and import status must work in the public product; the
  current public deployment's disabled linking flow must be replaced.
- A session can access only its owner's mail, attachments, conversations,
  workspaces, transcripts, artifacts, events, caches and SQL results. Derive
  ownership at every hop from authenticated server state. Inviting someone does
  not share the inviter's data. No cross-user conversations or battles are in
  scope; both battle branches belong to their initiating user.
- Replace the current single-owner public DB login with server-selected,
  tenant-specific access boundaries. Never route all invited users through
  `gmail_search_public` pinned to the original owner, and never relax its fixed
  policy into a caller-controlled setting. Separate identity/invitation metadata
  from mailbox access. The data gateway uses each owner's restricted credentials;
  the SQL gateway selects a dedicated read-only login for each owner.
- Use per-user queue, storage and spending quotas, plus system-wide limits, so
  one invitee cannot monopolize execution. Invitation removal revokes browser
  sessions and run leases, cancels ongoing work and prevents new credential use.
  Define mailbox disconnect, data deletion and backup retention explicitly.
- Application administrators manage admission and operations without automatic
  access to invitees' mail. Infrastructure operators remain within the stated
  trusted-host boundary; this design does not claim end-to-end secrecy from the
  machine's root administrator.

## Recommended architecture

Use a disposable **Firecracker microVM per agent run**, on a dedicated Linux
execution machine/VM that contains no mailbox database, home directories,
Docker control socket, OAuth refresh credentials or broker administration keys.
A microVM provides a separate guest kernel. Use Firecracker's production jailer,
seccomp and host configuration; a microVM alone is not a complete boundary.
Validate KVM/nested-virtualization availability before choosing the worker host.

Trusted components remain separate:

1. Public web/API authenticates the owner and owns conversation state.
2. A job controller starts a run using a fixed image and server-selected limits.
3. A capability gateway performs authenticated mail retrieval, model inference,
   attachment transfer, artifact publication and SQL queries on behalf of a run.
4. Workspace storage holds owner/conversation-scoped versions of files.
5. A SQL gateway compiles analytical queries against the existing database using
   a separate SELECT-only login per owner; bounded local extracts support additional analysis.

The guest reaches only its gateway through a constrained channel (e.g. a host
relay over vsock). No general IP route, DNS, LAN, cloud metadata, production DB,
or other guest access. Firewall and controller policy enforce this outside the
guest; an in-guest firewall is not an adequate boundary against arbitrary code.
The gateway is network reachable only from workers/trusted app components and
has no unscoped administrator routes on its guest-facing interface.

The controller accepts a run ID and fixed configuration, never model-supplied
container flags, volume paths, image names or executable host commands. Treat
its access to the VM launcher as a privileged API.

### Alternatives

- **gVisor/runsc per run on a dedicated worker VM:** a reasonable lower-effort
  option that reuses the current Docker images. It adds a userspace kernel
  boundary, but needs compatibility tests for both agent CLIs and the Python
  stack. Keep gVisor's isolated networking; do not enable host networking to
  make a failing dependency work. Choose this if the compatibility spike passes
  and its simpler operations outweigh the preference for a separate guest kernel.
- **Ordinary Docker containers on the mail host:** not the proposed public
  boundary. Removing shared mounts would help, but leaves arbitrary agent code
  next to sensitive services with the same host kernel.

Recommendation is Firecracker on a separate worker. Do a short runnable
compatibility spike before committing to the worker implementation; do not
quietly fall back to the present shared container if it fails.

## Capability contract

Every run receives short-lived capability credentials, never platform secrets.
Bind them to owner, conversation, run, allowed operations, expiry and a live
server-side lease. Validate both signature and active lease on every operation;
expiry alone does not provide immediate revocation. The existing MCP session
tokens provide a starting point, not the finished authorization boundary.
The gateway derives identity from the credential; submitted user/session/path
identifiers cannot expand it. Separate audiences for inference, retrieval and
artifact operations prevent token confusion.

A shell process may read its own run token. This is expected: security comes
from the token's limited authority, network boundary and prompt revocation,
not from trying to hide a credential from code in the same guest.

Provider API keys and Claude refresh credentials stay at the trusted inference
gateway. Allow fixed inference endpoints/models, bound spend and token counts,
and strip caller-supplied authorization/forwarding headers. Validate a strict
provider-specific request schema: disable hosted browsing, connectors, remote
image/file URL fetches and all other provider-executed network tools unless
separately authorized. Model/provider choices come from server-approved run
configuration, not a guest-selected upstream. Deny account APIs,
provider file stores, arbitrary upstream URLs and redirects. Disable unrelated
CLI telemetry/update calls. Test native Claude Code's authentication and
streaming behavior against this gateway; compatibility must be demonstrated,
not assumed. If it requires credential injection, redesign or use an equivalent
approved adapter; do not mount the host's credential directory as a shortcut.

The chosen model providers still receive the mail used for inference. This is
an intentional data recipient, not something VM isolation can prevent.

## Restoring each capability

| Capability | Implementation |
|---|---|
| Search, full threads, facts | Existing retrieval API behind the run gateway; owner derived server-side; preserve pagination and citations. |
| Shell, Python, read/write/edit | Run freely inside the disposable guest, with bounded CPU/RAM/processes/disk/time; no host execution or mounts. |
| Attachments | Guest requests an attachment ID; gateway checks owner and streams bounded bytes. No arbitrary filesystem path or URL supplied by the model. |
| Charts, CSV/XLSX, documents | Existing Python packages baked into versioned images; guest writes files and explicitly publishes them. |
| Model selection | Inference gateway supports the existing Gemini/Anthropic/OpenRouter model menu and enforces per-run budgets. |
| Pi/Claude workflows | Install agent runtimes and approved extensions in immutable images; subprocess/subagent execution remains inside the run boundary. |
| Battles | Two independent guests/capabilities from the same read-only conversation snapshot, separate output namespaces and combined budget; no concurrent writes to one session file. |
| Multi-turn workspaces | Versioned per-conversation data volume reattached to a fresh guest under an exclusive lease. Preserve documents/code but reset system, home, credentials and processes. |
| SQL/schema tools | Analytical SQL gateway with per-user read-only logins and bounded local extracts; no whole-mailbox copy. |

## Workspace and artifact rules

Keep guest disk images opaque to the trusted host: never mount a guest-controlled
filesystem in the host kernel. Extract files and process document previews in
separate disposable sandboxes. Only bounded validated byte streams cross into
trusted storage.

One active writer per conversation workspace. Assign a monotonically increasing
fencing number with each lease and commit by comparing against the current
workspace version and lease number. A stale worker cannot overwrite a newer
run even if it finishes later. Successful ordinary runs advance the version;
failed/cancelled changes are quarantined and never become the default workspace.
Battle branches keep separate versions; selecting a branch explicitly chooses
the next workspace, and an unselected battle leaves the base version unchanged.
On completion, preserve a bounded
workspace version; new runs boot a clean system and receive only that
conversation's files. Keep CLI conversation transcripts in a separate owner/conversation namespace and
restore only the selected transcript, not a shared `/sessions` directory or a
whole agent home. Battle branches get separate transcripts and never silently
merge their filesystem changes. Persisted files remain untrusted. Do not automatically
load workspace-controlled agent extensions, startup scripts, credentials or
configuration on the next turn. Explicitly executing a saved script is allowed
inside the new isolated guest.

A guest publishes bytes through a run-bound upload operation, not a path that
the host later resolves. This replaces the current shared-bind-mount artifact
reader. Enforce streaming byte quotas and server-selected object keys. Reject
symlinks/devices/path traversal when importing/exporting archives; never let a
race between path checking and file opening select a host file. MIME labels and
filenames are untrusted. Deliver active content as downloads or from a separate
cookieless origin with restrictive policy. Keep inline safe previews and charts
so security does not remove useful output.

## SQL without copying the database

**Revised requirement:** no full per-user database copies. Use the existing
PostgreSQL corpus behind a trusted SQL gateway, plus optional small task-specific
extracts inside each agent VM. Storage growth is workspace/output data, not a
second mailbox. This replaces the earlier analytics-copy proposal.

### Recommended: per-user readers plus a constrained SQL compiler

1. Provision a dedicated SELECT-only login per invited user, with fixed
   restrictive row policies for that owner. It owns no tables, has no privileged
   memberships or BYPASSRLS, cannot create schema objects and cannot change the
   mapping to its owner. Connect directly as that login; never start as a
   superuser and then SET ROLE. Keep pools separated by login identity.
2. Only the trusted gateway holds these passwords and chooses the login from
   the authenticated run capability. The agent VM has no PostgreSQL route or
   database credential. Do not reuse the public application role, which can
   write conversation/session records.
3. Expose a documented analytical SQL language: SELECT, joins among allowed
   mailbox tables, filters, grouping, aggregates, approved window functions,
   nonrecursive CTEs and subqueries. Parse the entire request into a supported
   syntax tree and compile a fresh parameterized statement. Reject unsupported
   syntax at every nesting level; never pass the original SQL through after a
   blacklist check. Resolve/qualify all relations, columns, functions, types and
   operators against a closed schema. No system catalogs, arbitrary functions,
   user-defined operators/casts, DDL, DML, COPY or setting/role commands.
4. RLS remains the independent data boundary if the compiler accidentally
   omits a tenant filter. Test view/function ownership and policy composition;
   a security-definer view must not silently bypass that boundary. Schema
   discovery returns the same explicit analytical schema, not raw catalogs.
5. Apply read-only transactions, timeouts, memory/temp-space limits, bounded
   results, per-user concurrency and cancellation. An output-row cap alone
   does not bound the work needed by an aggregate or join. Keep BM25 in the
   established parameterized search tool initially; add it to analytical SQL
   only after extension/RLS/performance qualification.

This supports live whole-mailbox counts, joins and analysis without copies.
It restores the analytical SQL capability, not unrestricted PostgreSQL
administration or arbitrary extension execution. The compiler is a real security
component and needs adversarial tests and independent review.

### Complement: bounded local analysis extracts

For queries outside the supported SQL language, fetch only the needed,
owner-authorized columns/rows and load them into DuckDB or Python inside the
agent VM. Arbitrary local SQL/code then touches only that run's data. Enforce
transfer/storage budgets and record completeness. A limited search-result sample
must never be presented as an exact whole-mailbox count; compute global
aggregates through the gateway before selecting rows for detailed analysis.

### Other option: one shared read replica

A shared replica could later keep analytical load off ingestion without a copy
per user. It still costs one database copy, retains the per-user role/compiler
requirements and introduces replication lag. It is optional, not a prerequisite
for the proposed no-copy version.

### Changed risk and acceptance criteria

The database engine and its approved extensions now remain a shared trusted
component. This design does not contain a PostgreSQL/extension compromise to
one owner's copied database, as the more expensive proposal would. VM isolation
still contains agent shell code; it does not isolate database-server execution.
Qualify the query language and performance on the installed engine, test malicious
nested SQL and function/cast/operator tricks, and prove foreign rows cannot be
read through any allowed relation. Test that one user's expensive query cannot
exhaust the shared service. Do not enable raw SQL while the compiler is unfinished.

## Internet access and prompt injection

Restore the actual existing toolset; the current Pi prompt already describes
general Internet access and package installation as unavailable. Preinstall the
common analysis packages and use an image-build process for additions.

If web research/downloads are required, provide a separate network worker with
no mailbox, workspace or mail credentials. Requests derived from private mail
must still cross a deliberate policy/approval boundary: separating workers
alone does not prevent leaking mail in a search query or URL. An owner-approved
outbound request must show its destination and content; checking only the domain
is insufficient. Enforce redirect and resolved-IP checks at each hop.

Unrestricted unattended networking plus arbitrary code that can read mail is
incompatible with a promise that mail cannot be exfiltrated. State this limit
explicitly rather than claim that a sandbox fixes it. Rendering model output
must not create a backdoor: no automatic external Markdown images or embedded
resources, and deliberate navigation for external links.

## Reliability and lifecycle

Use a durable job/lease registry. Each run has an idempotency key, bounded queue,
hard deadline, heartbeat and resource/spend quotas. Reserve shared budgets
atomically across battle branches and subagent calls so concurrency cannot
exceed the total. Revoke capabilities first
on cancellation or failure, propagate cancellation to active inference streams
and SQL queries, terminate the guest, then finalize workspace state. Revalidate
long-running streams at bounded intervals and check current authorization and
fencing again at artifact/workspace commit. Revocation cannot recall bytes
already sent to an authorized model provider, but it must prevent subsequent
operations and publication.
A sweeper outside the request process removes orphaned guests after crashes.
Test controller restarts, worker loss, expiry during tool calls and reconnects.
Persist streamed events outside the guest and support replay by event cursor.
Never reuse a partially failed run's credentials or running processes.

Warm only clean images/VMs; never hand another run a warmed instance containing
mail, workspace data, tokens or agent history. Monitor failure and quota events
without recording credentials or unnecessary mail content. Keep tested backup
and restore procedures for conversation workspaces and the existing database.

## Delivery and acceptance

1. **Compatibility spike:** Pi and native Claude run in a clean microVM through
   the inference gateway; Python produces a chart and spreadsheet; streaming,
   cancellation and tool calls work. Demonstrate no model-provider keys in guest.
2. **Full-tool candidate:** scoped gateway, isolated run/workspace lifecycle,
   safe artifact publication, model menu/workflows/battles and the no-copy analytical
   SQL gateway, invitation management and account-bound Gmail connection/onboarding.
   Exercise realistic email tasks against synthetic data with exact expected
   results, including long-running analysis and conversation continuation.
3. **Adversarial qualification and rollout:** independent review; hostile mail
   attempts credential theft, other-workspace access, identity forgery, SQL role
   escalation, network exfiltration, symlink/archive attacks and process escape.
   Check direct network/DNS/IPv6/metadata denial, stale-token rejection, host
   restart cleanup, quotas and cross-user isolation with two synthetic owners.
   Verify private output cannot be loaded by another browser session. Exercise
   two independently invited Google accounts, wrong-account consent, invitation
   replay/revocation, guessed object IDs and cache reuse. Each must see only
   its own seeded mailbox through every tool and browser route, including after
   restarts and simultaneous battles. Application admin status must not bypass
   these checks.

No success claim based only on a sandbox starting. Acceptance requires both
functional parity and boundary tests. Switch the public app only after those
checks. Rollback disables the replacement ingress or restores a previously
qualified multi-user release. The current owner-only deployment can remain
available to its original owner, but cannot serve as a fallback for invitees.
Do not
restore broad shared-container access during an incident.

## Initial implementation map

- `runtime_pi.py` / `pi_protocol.py`: replace shared docker-exec startup with a
  job-controller interface; adapt Claude runtime through the same interface.
- New controller/worker package: fixed VM images, leases, lifecycle and quotas.
- `mcp_tools_server.py`: dedicated guest-facing capability gateway, online lease
  checks, no guest-visible administrator routes; retain trusted control API separately.
- New inference proxy and constrained SQL compiler/gateway, per-user reader
  provisioning and bounded local analysis extracts.
- Auth/broker onboarding: active invitations, identity-bound Gmail linking,
  revocation and tenant provisioning; replace single-owner startup assumptions
  without weakening the existing deployment's fixed database boundary.
- Replace shared-path publishing with byte upload and versioned workspace storage.
- Web model/battle/attachment/artifact flows: restore feature presentation only
  when corresponding capabilities are available and verified.

This is a multi-component infrastructure change, not a safe one-flag reversal.
Main uncertainties to resolve in the spike: worker/KVM placement, native Claude
proxy compatibility, SQL compiler coverage and shared-database query cost and large-attachment performance.
No implementation or service changes are authorized by this proposal alone.

## References

- [Firecracker production host setup](https://github.com/firecracker-microvm/firecracker/blob/main/docs/prod-host-setup.md): production isolation depends on host/jailer configuration, not merely starting a VM.
- [Firecracker getting started](https://github.com/firecracker-microvm/firecracker/blob/main/docs/getting-started.md): guest kernel/root filesystem and production jailer requirements.
- [gVisor security model](https://gvisor.dev/docs/architecture_guide/security/): the sandbox does not replace application architecture and permitted resource access must still be constrained.
- [gVisor networking](https://gvisor.dev/docs/architecture_guide/networking/): isolated userspace networking and the host-networking tradeoff.
- [PostgreSQL row security](https://www.postgresql.org/docs/17/ddl-rowsecurity.html): policies complement privileges; privileged roles and policy composition require care.
