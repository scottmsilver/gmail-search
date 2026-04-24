# Pinned Python sandbox image for the deep-analysis Analyst agent.
#
# Runs the LLM-authored analysis snippet against a pre-seeded
# pandas DataFrame + a read-only Postgres connection. Every design
# choice below is about reducing what a malicious (or buggy) snippet
# can do:
#
#   - Pinned Python + pinned deps: no live package installs at run
#     time. What's baked into the image is the full toolkit.
#   - Non-root user (uid 1000): can't touch /usr/bin, /etc.
#   - Read-only root filesystem: enforced at `docker run --read-only`
#     time by the caller; the dir structure is ready for it.
#   - Workdir is /work (tmpfs mount from caller); the only place the
#     snippet can write.
#
# Network / memory / CPU caps are applied by the caller via
# `docker run` flags, not inside the image, so they're tunable
# without rebuilds.

FROM python:3.12-slim-bookworm

# Matplotlib needs a writable config dir and a no-GUI backend; we
# give it both at the image level so the snippet doesn't have to.
ENV MPLCONFIGDIR=/work/.mpl \
    MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 1000 analyst \
 && mkdir -p /work \
 && chown analyst:analyst /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Preamble the sandbox injects before every snippet: imports the
# standard stack, loads the seeded evidence DataFrame, exposes a
# `save_artifact` helper, and opens the read-only DB connection.
# The caller writes it to /work alongside the snippet.
COPY preamble.py /opt/analyst/preamble.py

USER analyst
WORKDIR /work
