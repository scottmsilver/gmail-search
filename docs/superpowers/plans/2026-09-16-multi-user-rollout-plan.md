# Getting to a deployed multi-user-safe version

Written 2026-09-16. Supersedes nothing; sits above
[the migration plan](2026-09-15-migration-after-ctid-fix.md), which covers one
track of this in detail.

**The honest summary: the migration is nearly ready, the multi-user application
is not.** The database work has been rehearsed end to end at production scale.
What has not been built or provisioned is the runtime that multiple users would
actually reach — the isolated worker, the Gmail broker, and the invited service
on the host. Two of those need your cloud access and cannot be done unattended.

## Where we actually are

Verified today, not inferred:

| | Status |
| --- | --- |
| pg_search ctid bug | Root-caused, patched, **deployed to live** |
| Live engine | PostgreSQL 16.15, pg_search 0.23.0, healthy |
| Live schema | Unpartitioned, TEXT-shaped (`key_field=id`), 3 owners, 443k messages |
| Migration mechanism | Rehearsed at full scale: ~4 min work, <1 GiB, ~2.2 GiB WAL |
| Rollback | Rehearsed: dump 8.1 min, restore 6.9 min, verified 4 ways |
| Schema profile binding | Done — selected, carried, verified at connect |
| BM25 callers | Realigned to the bound key; mismatches now fail loudly |

## Four tracks

### Track A — Make the application owner-safe (no migration yet)

This is the track that decides whether "multi-user safe" is a claim we can make.

- **A1. Fix the gateway failures and the hang.** 25 failures and one test that
  hangs while holding `ACCESS EXCLUSIVE`, all pre-existing and reproduced on the
  unpatched engine. One is `test_token_selects_only_its_owners_rows` — owner
  isolation is the product's whole premise, so this ranks first. The hang also
  makes the suite unrunnable as a single invocation, which is how it has been
  eating hours. Effort: unknown until diagnosed; that uncertainty is the point.
- **A2. Finish the caller inventory.** Three of ten items done. Remaining: the
  SQL examples handed to the model still name `search_id`; the owner-qualified
  call sites; and the **TEXT installer** — fresh installs currently build the
  numeric shape while live is TEXT, a divergence the profile binding now states
  out loud instead of hiding.
- **A3. Give the migration scripts a production mode.** They refuse any DSN that
  is not port 55440 on a disposable database — deliberately. Production needs an
  explicit, separately guarded path, not a deleted check.

Unattended. Depends on nothing. **Start here.**

### Track B — Migrate the live schema

- **B1.** Fresh verified backup immediately before (the cold-copy procedure is
  rehearsed and recorded).
- **B2.** Run phase one then phase two. Measured at ~4 minutes of work with
  ~94 s of exclusive locking in phase two and ~43 s in phase one.
- **B3.** Deploy owner-aware callers **in the same window**. Readers and writers
  must move together; the profile binding turns a mistake here into a refused
  connection rather than a silent wrong-shape query.

Depends on A2 and A3. Needs your go-ahead for the window.

### Track C — The multi-user runtime

This is the long pole, and most of it needs you.

- **C1. A dedicated worker host.** No enrolled production VM exists. Needs a
  machine, then enrolment and a verified export of the pinned full-agent image.
  The installer and manager unit are written.
- **C2. The Gmail broker.** `gmailBrokerV1` does not exist in
  `silver-oauth-broker`, and none of the six `GMAIL_V1_*` secrets are created.
  Needs a dedicated OAuth Web client with redirect
  `https://auth.oursilverfamily.com/v1/gmail/callback`, the secrets in Secret
  Manager, Firestore TTL on `gmail_v1_metadata.expires_at`, and the Hosting
  rewrite. **I cannot create these for you** — they are credentials in your
  cloud account.
- **C3. Install the invited service.** `invited_server.py` is a real entrypoint
  and the units exist under `deploy/public/`, but no `gmail-search-invited-*`
  unit is installed on this host. Live still runs the old owner-only
  `public_server`.

C1 and C2 are independent of Tracks A and B and can proceed in parallel the
moment you have time for the cloud console.

### Track D — Qualification and cutover

- **D1.** Real Google sign-in and separate Gmail consent with two invited
  accounts.
- **D2.** Adversarial isolation review with two real mailboxes — the check that
  A1's unit tests only approximate.
- **D3.** Acceptance suite, restart and fault injection, concurrent load.
- **D4.** Cutover. Rollback disables invitee ingress; it never routes invitees
  into the owner-only app.

Depends on everything above.

## The critical path

```
A1 ──► A2 ──► A3 ──► B1 ──► B2/B3 ──┐
                                     ├──► D1 ──► D2 ──► D3 ──► D4
C1 ──────────────────────────────────┤
C2 ──────────────────────────────────┘
        (C3 after C1 + C2)
```

A and C are independent. **C2 is the item most likely to set the date**, because
it is gated on your availability rather than on engineering, and D1 cannot start
without it.

## Division of labour

**I can do unattended:** A1, A2, A3, B1, the C1/C3 software preparation, and
writing D3's acceptance suite.

**Needs you:**
- C2 — OAuth client and six secrets. Nobody else can create them.
- C1 — a machine to dedicate as the worker.
- B2 — approval for the live migration window.
- D1/D2 — two real Google accounts, and a human completing consent.

**Needs a decision, not work:** whether the remaining gateway failures block the
migration or merely block the multi-user launch. My reading is that they block
the launch but not the migration, since the migration's own suites pass and the
gateway is not on the live read path today.

## What I would do next

A1. Not because it is the most fun, but because a failing test named
`test_token_selects_only_its_owners_rows` is the one result that would make
every other green tick meaningless if it turns out to be real.
