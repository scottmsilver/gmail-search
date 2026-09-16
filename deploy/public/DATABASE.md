# Owner-only public database login

`provision_database.py` creates the dedicated `gmail_search_public` login for a
single **existing** `users.id`. The public app must separately allow only that
owner's verified identity. Use this role's DSN only in the public backend; keep
private ingestion on its existing login.

Run using the project virtual environment. Supply the administrator connection
string through `GMAIL_PROVISION_DSN`, loaded from a private environment file.
Do not put passwords in shell command arguments or committed files.

```sh
.venv/bin/python deploy/public/provision_database.py --owner-id OWNER_USER_ID
```

This performs catalog preflight and prints password-free SQL without changing
the database. Review the SQL before applying:

```sh
.venv/bin/python deploy/public/provision_database.py \
  --owner-id OWNER_USER_ID --apply \
  --password-file /absolute/private/directory/gmail-search-public.password
```

The directory must already exist and should have mode 0700. Apply creates a
random password in a mode-0600 file and reuses that file on subsequent runs.
It never prints the password. An unsuccessful database transaction rolls back;
a newly created password file remains available for retry. Provisioning a
different owner requires an intentional administrator action; never accept the
owner ID or connection string from an HTTP request.

## Database boundary

- Login has no superuser, database creation, role creation, replication, or RLS
  bypass capability. Preflight rejects role memberships and object ownership.
- Mail, summaries, embeddings, facts, index pointers, users and artifacts are
  read-only. Conversations/messages can be edited/deleted; sessions can be
  inserted/updated; events and costs can only be appended. Sequence access is
  limited to these inserts. Users cannot be created/updated by this login.
- Every granted table has a fixed owner predicate in both a permissive and a
  **restrictive** policy specific to this role. Existing policies reading
  `app.user_id` cannot bypass the fixed owner predicate, even after `set_config`.
  Child messages/events/artifacts are scoped through their owning parent.
- Tables gaining RLS preserve their preexisting non-superuser access through
  policies naming the catalog-discovered roles explicitly. This includes the
  legacy analyst's existing embedding access. No new grants go to those roles.
  Tables already using RLS retain their existing policies.
- Global query cache, jobs, sync state, invitations and OAuth state receive no
  grants. The public launcher must bypass schema initialization and job reaping;
  public search must avoid persistent global query-cache reads/writes.

The effective-permission audit aborts application if `PUBLIC` gives schema or
database CREATE, unexpected table access, or access to a non-system
`SECURITY DEFINER` function. Administrator review is required to remove those
ambient grants; the script does not change grants for `PUBLIC`. Standard
PostgreSQL temporary-table and system-catalog access may remain. This role must
never receive arbitrary SQL, code execution or filesystem tools through the
public runtime. RLS does not prevent PostgreSQL constraint-error side channels.

## Validation before serving

Test on an isolated database first. Verify owner reads and chat writes succeed;
other-user reads return zero rows even after changing `app.user_id`; cross-owner
inserts/updates, mail writes, role switching and schema creation fail. Verify the
real search implementation and installed BM25 extension under RLS before using
this role in production. Check owner sign-in, retrieval, chat completion and
cancellation using the candidate public backend before ingress.

No live provisioning is performed by the unit tests. They validate generated
policies/grants, SQL quoting, and password-file permissions/idempotence.

## Installed extension ACL adjustment

The live preflight found ambient PUBLIC CREATE on `paradedb` and `pdb`, ALL
on `paradedb._typmod_cache`, public query-statistics views, and execution of
`paradedb._save_typmod(text[])`. A broad ACL rewrite was rejected by automatic
approval review. The applied alternative in `restrict-extension-acls.sql`
changes only these named objects and preserves the audited existing roles'
access explicitly. Read-only spatial/index metadata remains allowed by the
explicit `EXTENSION_METADATA` list. No mail-table contents were changed.

Live verification as the new role confirmed owner rows visible, foreign rows
hidden even after changing `app.user_id`, SET ROLE to privileged login denied,
and global `query_cache` inaccessible. Public search skips that shared cache.
The separate public launcher checks the non-superuser role and its visible
owner against the public admission allowlist at every start.
