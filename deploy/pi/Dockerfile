# Pi agent harness sandbox for gmail-search deep mode.
# The whole `pi` process runs in here; the container boundary is the
# only thing between model-driven bash and the host.
FROM node:24-bookworm-slim

ARG PI_VERSION=0.84.4

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       bash ca-certificates git ripgrep procps \
       python3 python3-pandas python3-matplotlib python3-numpy \
  && rm -rf /var/lib/apt/lists/*

RUN npm install -g --ignore-scripts "@earendil-works/pi-coding-agent@${PI_VERSION}"

COPY extensions/gmail-tools/package.json extensions/gmail-tools/package-lock.json /opt/gmail-tools/
RUN cd /opt/gmail-tools && npm ci --ignore-scripts --omit=dev
COPY extensions/gmail-tools/*.ts /opt/gmail-tools/

# node:24 ships a `node` user at uid 1000; rename it so the workspace
# files it writes are owned by the host user (also uid 1000).
RUN usermod -l pi -d /home/pi -m node && groupmod -n pi node \
  && mkdir -p /home/pi/.pi/agent /sessions /workspaces \
  && chown -R pi:pi /home/pi /sessions /workspaces

ENV MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
USER pi
WORKDIR /workspaces
ENTRYPOINT ["sleep", "infinity"]
