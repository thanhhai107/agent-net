FROM python:3.12-slim

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/nika/src" \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20.x (includes npm)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Codex CLI: official installer when it succeeds, otherwise npm
RUN if curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/codex-install.sh && \
        CODEX_NON_INTERACTIVE=1 CODEX_INSTALL_DIR=/usr/local/bin sh /tmp/codex-install.sh; then \
        echo "installed codex via official installer"; \
    else \
        npm install -g @openai/codex || true; \
    fi

RUN npm install -g @anthropic-ai/claude-code

# Agent runtime + SDK packages
RUN uv pip install --system --prerelease=allow \
    "claude-agent-sdk" \
    "openai-codex>=0.1.0b3" \
    "langgraph" \
    "pydantic>=2.11,<2.14" \
    "python-dotenv" \
    "langchain-core"

RUN if ! command -v codex >/dev/null 2>&1; then \
        CODEX_BIN="$(python -c "from importlib.resources import files; print(files('codex_cli_bin') / 'bin' / 'codex')")" && \
        ln -sf "$CODEX_BIN" /usr/local/bin/codex; \
    fi

COPY src/agent /nika/src/agent
COPY src/nika /nika/src/nika

RUN useradd -m -u 1000 -s /bin/bash agent && \
    mkdir -p /workspace /home/agent/.codex && \
    chown -R agent:agent /workspace /home/agent /nika

USER agent
WORKDIR /workspace
