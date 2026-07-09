FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PATH="/usr/local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        gnupg \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20.x (includes npm) without Debian's heavy npm dependency tree
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# uv via pip
RUN pip install --no-cache-dir uv

# Codex CLI: official installer when it succeeds, otherwise npm
RUN if curl -fsSL https://chatgpt.com/codex/install.sh -o /tmp/codex-install.sh && \
        CODEX_NON_INTERACTIVE=1 CODEX_INSTALL_DIR=/usr/local/bin sh /tmp/codex-install.sh; then \
        echo "installed codex via official installer"; \
    else \
        npm install -g @openai/codex || true; \
    fi

# Claude CLI
RUN npm install -g @anthropic-ai/claude-code

# SDK packages aligned with pyproject.toml [sdk] extra
RUN uv pip install --system --prerelease=allow \
    "claude-agent-sdk" \
    "openai-codex>=0.1.0b3"

# Ensure codex is on PATH (npm install or SDK-bundled binary)
RUN if ! command -v codex >/dev/null 2>&1; then \
        CODEX_BIN="$(python -c "from importlib.resources import files; print(files('codex_cli_bin') / 'bin' / 'codex')")" && \
        ln -sf "$CODEX_BIN" /usr/local/bin/codex; \
    fi

RUN useradd -m -u 1000 -s /bin/bash agent && \
    mkdir -p /workspace && \
    chown agent:agent /workspace

USER agent
WORKDIR /workspace
