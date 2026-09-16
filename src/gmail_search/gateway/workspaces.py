"""Fenced metadata for opaque workspace bytes; no host filesystem mounting.

Only a qualified guest importer may decode a snapshot. It must restore data to
the workspace, never an agent home, startup configuration, or credential store.
Browser branch selection requires a server-authenticated owner supplied by the
trusted application. This library is not mounted in the public application.
"""
from .registry import AccessDenied


class Workspaces:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.capabilities = artifacts.capabilities
        self.registry = artifacts.registry
        with self.registry._transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS workspace_versions (
                owner_id TEXT, conversation_id TEXT, version INTEGER, artifact_id TEXT NOT NULL,
                PRIMARY KEY(owner_id,conversation_id,version))''')
            db.execute('''CREATE TABLE IF NOT EXISTS workspace_branches (
                run_id TEXT PRIMARY KEY, artifact_id TEXT NOT NULL)''')

    def _artifact(self, db, run, artifact_id):
        row = self.artifacts._metadata(db, run['owner_id'], run['conversation_id'], artifact_id)
        if row['run_id'] != run['run_id']:
            raise AccessDenied()
        return row

    def _version(self, db, run, artifact_id):
        version = run['workspace_version'] + 1
        db.execute('INSERT INTO workspace_versions VALUES(?,?,?,?)',
                   (run['owner_id'], run['conversation_id'], version, artifact_id))
        db.execute('UPDATE conversations SET version=? WHERE owner_id=? AND conversation_id=?',
                   (version, run['owner_id'], run['conversation_id']))
        return version

    def commit(self, token, artifact_id):
        """Atomically promote the current writer's uploaded snapshot and finish it."""
        with self.registry._transaction() as db:
            run = self.capabilities._authorize(db, token, 'artifact', 'workspace.commit')
            self._artifact(db, run, artifact_id)
            version = self._version(db, run, artifact_id)
            self.registry._finish(db, run['run_id'], 'completed')
            return version

    def restore(self, token):
        """Return only the immutable base version bound to this live run."""
        with self.registry._transaction() as db:
            run = self.capabilities._authorize(db, token, 'artifact', 'workspace.read')
            if run['workspace_version'] == 0:
                return None
            row = db.execute('SELECT artifact_id FROM workspace_versions WHERE owner_id=? AND conversation_id=? AND version=?',
                             (run['owner_id'], run['conversation_id'], run['workspace_version'])).fetchone()
            if not row:
                raise AccessDenied()
        data = self.artifacts.read(run['owner_id'], run['conversation_id'], row['artifact_id'])
        self.capabilities.authorize(token, audience='artifact', operation='workspace.read')
        return data

    def stage_branch(self, token, artifact_id):
        """A battle run may preserve a candidate without changing the base version."""
        with self.registry._transaction() as db:
            run = self.capabilities._authorize(db, token, 'artifact', 'artifact.commit')
            if run['writer']:
                raise AccessDenied()
            self._artifact(db, run, artifact_id)
            old = db.execute('SELECT artifact_id FROM workspace_branches WHERE run_id=?', (run['run_id'],)).fetchone()
            if old and old['artifact_id'] != artifact_id:
                raise AccessDenied()
            db.execute('INSERT OR IGNORE INTO workspace_branches VALUES(?,?)', (run['run_id'], artifact_id))

    def select_branch(self, owner_id, conversation_id, run_id):
        """Trusted authenticated UI action: choose one completed battle snapshot.

        No administrator override. A changed base version, active writer, stale
        fence, failed branch or foreign conversation is rejected atomically.
        """
        with self.registry._transaction() as db:
            self.registry._owner(owner_id)
            run = db.execute('SELECT * FROM runs WHERE run_id=? AND owner_id=? AND conversation_id=?',
                             (run_id, owner_id, conversation_id)).fetchone()
            if not run or run['writer'] or run['status'] != 'completed':
                raise AccessDenied()
            current = db.execute('SELECT * FROM conversations WHERE owner_id=? AND conversation_id=?',
                                 (owner_id, conversation_id)).fetchone()
            if current['writer'] is not None:
                raise AccessDenied()
            self.registry._fence(db, run)
            branch = db.execute('SELECT artifact_id FROM workspace_branches WHERE run_id=?', (run_id,)).fetchone()
            if not branch:
                raise AccessDenied()
            self._artifact(db, run, branch['artifact_id'])
            version = self._version(db, run, branch['artifact_id'])
            db.execute('UPDATE conversations SET fence=fence+1 WHERE owner_id=? AND conversation_id=?', (owner_id, conversation_id))
            return version
