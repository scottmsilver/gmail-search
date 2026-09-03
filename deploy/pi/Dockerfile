# Pi agent harness sandbox for gmail-search deep mode.
# The whole `pi` process runs in here; the container boundary is the
# only thing between model-driven bash and the host.
FROM node:24-bookworm-slim

ARG PI_VERSION=0.84.4

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       bash ca-certificates git ripgrep procps curl jq \
       python3 python3-pandas python3-matplotlib python3-numpy \
       python3-openpyxl python3-docx python3-pypdf python3-requests \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}"

# pi-mcp-adapter gives pi direct MCP access to the gmail-search tools server;
# it replaces the hand-rolled gmail-tools bridge extension. typebox is a hard
# runtime import of the adapter despite being listed as an optional peer dep;
# @earendil-works/pi-ai and pi-tui are pinned to match PI_VERSION above (kept
# in sync by hand — they don't read the ARG since npm ci resolves strictly
# from the lockfile). Versions and their full resolved dependency graph live
# in pi-pkgs/package.json + package-lock.json (committed) so the build is
# reproducible instead of re-resolving npm's graph on every image build;
# regenerate the lockfile with `npm install --ignore-scripts
# --package-lock-only` in that directory after bumping a version there.
COPY pi-pkgs/package.json pi-pkgs/package-lock.json /opt/pi-pkgs/
RUN cd /opt/pi-pkgs && npm ci --ignore-scripts
COPY mcp.json /opt/gmail-mcp.json

# node:24 ships a `node` user at uid 1000; rename it so the workspace
# files it writes are owned by the host user (also uid 1000).
RUN usermod -l pi -d /home/pi -m node && groupmod -n pi node \
  && mkdir -p /home/pi/.pi/agent /sessions /workspaces \
  && chown -R pi:pi /home/pi /sessions /workspaces

ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
USER pi
WORKDIR /workspaces
ENTRYPOINT ["sleep", "infinity"]
