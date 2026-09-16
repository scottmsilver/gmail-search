# Google auth setup — everything I need from you

Written 2026-09-16. This is the complete list of things only you can do,
because they are credentials in **your** Google Cloud account. Nothing here
needs me; everything here blocks the first real invited sign-in (track D1).

Budget about 30 minutes. All of it happens in the Firebase/Google Cloud project
**`silver-oauth-broker`** and in `~/development/silver-oauth/broker-firebase`.

## The shape of it

There are **two separate auth flows**, and keeping them separate is the point.

| | What it does | OAuth client | Status |
| --- | --- | --- | --- |
| **Sign-in** (`identityBroker`) | Proves who the person is | Reuses the existing one | Code extended, 2 values missing |
| **Gmail consent** (`gmailBrokerV1`) | Grants read access to one mailbox | **Needs a brand-new one** | Code written, 6 secrets missing |

A compromise of sign-in must not reach anyone's mail, and a compromise of the
mail grant must not be able to mint logins. That is why the Gmail flow gets its
own OAuth client and its own six secrets rather than borrowing the existing
ones. Do not reuse the `broker` or `identityBroker` client for step 2.

---

## Part 1 — Sign-in (the quick half)

`identityBroker` already exists and already works for wezterm. I extended it to
route a second app, `gmail-search`, to its own return URL and its own handoff
secret. Two values are missing.

### 1.1 Create the handoff secret

```sh
cd ~/development/silver-oauth/broker-firebase
openssl rand -base64 48 | tr -d '\n=' | \
  firebase functions:secrets:set GMAIL_SEARCH_IDENTITY_HANDOFF_SECRET --data-file -
```

**Write this value down.** It is one of three values that leave the project.
It becomes `GMS_IDENTITY_HANDOFF_SECRET` in the app's auth env file.

### 1.2 Set the return URL

This one is not a secret — it is a deploy-time parameter.

```
GMAIL_SEARCH_IDENTITY_RETURN_URL=https://gms.oursilverfamily.com/api/auth/callback
```

Put it in `broker-firebase/.env` (or `.env.silver-oauth-broker`) next to the
existing `WEZTERM_IDENTITY_RETURN_URL`. Its default is the empty string, so if
you skip this the gmail-search sign-in returns nowhere and fails silently.

No new OAuth client, no consent-screen change: sign-in only asks for identity,
and the existing client already has that scope.

---

## Part 2 — Gmail consent (the real work)

### 2.1 Enable the Gmail API

Google Cloud console → project `silver-oauth-broker` → APIs & Services →
Library → **Gmail API** → Enable. Skip if already on.

### 2.2 Create a dedicated OAuth client

APIs & Services → Credentials → **Create credentials** → OAuth client ID →
Application type **Web application**.

- Name it something you will recognise later, e.g. `gmail-search mailbox consent`.
- **Authorised redirect URI** — exactly this, one entry, no trailing slash:

  ```
  https://auth.oursilverfamily.com/v1/gmail/callback
  ```

- Leave Authorised JavaScript origins empty. The flow is server-side only.

Keep the client ID and client secret on screen for the next step.

### 2.3 Consent screen and scopes

Same project → OAuth consent screen. The three scopes the broker requests:

```
https://www.googleapis.com/auth/gmail.readonly
https://www.googleapis.com/auth/userinfo.email
openid
```

`gmail.readonly` is a **restricted** scope. While the app is in *Testing*, add
each invited address as a test user and it works immediately — this is the
right choice for two invited accounts. Publishing to *In production* with a
restricted scope triggers Google's security assessment, which takes weeks and
you do not need it. Stay in Testing.

> Testing mode expires refresh tokens after 7 days. For two known invitees that
> means re-consenting weekly. Worth knowing before you plan a demo around it.

### 2.4 Create the six secrets

Two come from the client you just made. Four you generate. Run all six from
`~/development/silver-oauth/broker-firebase`.

```sh
cd ~/development/silver-oauth/broker-firebase

# --- from the OAuth client in step 2.2 ---
firebase functions:secrets:set GMAIL_V1_GOOGLE_CLIENT_ID
firebase functions:secrets:set GMAIL_V1_GOOGLE_CLIENT_SECRET

# --- shared with the gmail-search app (record both) ---
openssl rand -base64 48 | tr -d '\n=' | \
  firebase functions:secrets:set GMAIL_V1_APP_BEARER --data-file -
openssl rand -base64 48 | tr -d '\n=' | \
  firebase functions:secrets:set GMAIL_V1_HANDOFF_SECRET --data-file -

# --- broker-internal, never leaves the project ---
openssl rand -base64 48 | tr -d '\n=' | \
  firebase functions:secrets:set GMAIL_V1_STATE_SECRET --data-file -

# --- keys AES-256-GCM: must decode to exactly 32 bytes, so no `tr -d` here ---
openssl rand -base64 32 | \
  firebase functions:secrets:set GMAIL_V1_TOKEN_ENC_KEY --data-file -
```

Three constraints the code enforces, each of which fails at runtime rather than
at `secrets:set`, so get them right now:

- `GMAIL_V1_TOKEN_ENC_KEY` must base64-decode to **exactly 32 bytes**. The
  `tr -d '\n='` used on the others would strip the padding and break it — note
  it is deliberately absent from that last command.
- `GMAIL_V1_APP_BEARER` and `GMAIL_V1_HANDOFF_SECRET` must each be at least 32
  printable ASCII characters and must **differ from each other**. The app-side
  validator rejects both otherwise.
- The two non-secret defaults are already compiled in and need no action unless
  a hostname changes: `GMAIL_V1_PUBLIC_ORIGIN=https://auth.oursilverfamily.com`
  and `GMAIL_V1_APP_CALLBACK=https://gms.oursilverfamily.com/api/auth/gmail-callback`.

### 2.5 Firestore TTL on consent tickets

Tickets live ten minutes in `gmail_v1_metadata`. Without a TTL policy, spent
ones accumulate forever.

```sh
gcloud firestore fields ttls update expires_at \
  --collection-group=gmail_v1_metadata --enable-ttl --project=silver-oauth-broker
```

Stored credentials live in `gmail_v1_credentials` and are deliberately **not**
TTL'd — `/v1/gmail/disconnect` deletes them.

---

## Part 3 — Deploy

> **Read this before running it.** The repo has uncommitted changes to
> `identity.ts`, `identity-flow.ts`, `identity-dispatch.ts` and their tests —
> that is the Part 1 extension. Deploying ships those too. Review the diff
> first; the identity flow is live for wezterm today.

```sh
cd ~/development/silver-oauth/broker-firebase
npx tsc --noEmit                    # passes as of 2026-09-16
firebase deploy --only functions:gmailBrokerV1,functions:identityBroker,hosting \
  --project silver-oauth-broker
```

`hosting` matters: the `/v1/gmail/**` rewrite must reach the new function
rather than the `**` catch-all, and rewrite order decides that.

### Verify the routing

```sh
curl -si https://auth.oursilverfamily.com/v1/gmail/consents \
  -X POST -H 'Content-Type: application/json' -d '{}' | head -1
```

- **`401`** — correct. `gmailBrokerV1` answered and rejected an unauthenticated
  call.
- **`200`, or any HTML** — the `**` rewrite won. The function is not reachable;
  fix the ordering in `firebase.json` before going further.

---

## What comes back to me

Exactly three values leave the project. Everything else stays in Secret Manager.

| Value | Where it lands |
| --- | --- |
| `GMAIL_SEARCH_IDENTITY_HANDOFF_SECRET` | `GMS_IDENTITY_HANDOFF_SECRET` in the invited auth env |
| `GMAIL_V1_APP_BEARER` | `broker.bearer` in `invited-runtime.json` |
| `GMAIL_V1_HANDOFF_SECRET` | `broker.signing_secret` in `invited-runtime.json` |

Send them however you normally move secrets. I will not print them back to you
and they do not belong in git — both target files are mode 0600 private config.

---

## Honest status

`gmailBrokerV1` is 323 lines written against the app's client contract and
type-checked. **It has never run against Google.** The first real sign-in is the
test, and I expect to iterate on the consent screen and scope approval before it
works end to end. Two behaviours worth knowing are already built in:

- The callback compares Google's `sub` against the subject the invitation was
  issued for and **refuses before storing anything** if they differ. Someone
  already signed in as a different Google account cannot attach that mailbox to
  someone else's invitation.
- The handoff JWT lives 45 seconds; the app rejects anything over 60. If clocks
  drift more than that between Cloud Functions and this host, consent fails with
  an expiry error rather than anything more descriptive.

Two constraints that are easy to break in a later edit: the client rejects any
response that is not `identity` content-encoding, so never put compression
middleware in front of this function; and the four endpoint response shapes are
matched exactly, so any extra field reads as "broker unavailable".

Detail on the wire contract and the endpoint table lives in
`~/development/silver-oauth/broker-firebase/GMAIL_V1_SETUP.md`.
