# GCP VM Setup Runbook

This runbook records the minimal steps to bring up NIKA on a fresh GCP VM and
verify that Studio/benchmark runs can start.

## 1. Install System Packages

```bash
sudo apt update
sudo apt install -y git curl wget build-essential docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"
newgrp docker
```

Check Docker:

```bash
docker ps
```

An empty table is fine. It means Docker is running and no containers are active.

## 2. Install uv And Python

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv python install 3.12
```

## 3. Clone And Install Project

```bash
cd ~
git clone <REPO_URL> agent-net
cd ~/agent-net
source ~/.local/bin/env
uv sync --python 3.12
uv run nika --help
```

`uv sync` must run from the project root, where `pyproject.toml` exists.

## 4. Configure Environment

```bash
cd ~/agent-net
cp .env.example .env
nano .env
```

For NetMind/custom OpenAI-compatible usage:

```env
MODEL_PROVIDER=custom
CUSTOM_API_URL=https://stream-netmind.viettel.vn/gateway/v1
CUSTOM_API_KEY=NetMind!@#
CUSTOM_MODEL=openai/gpt-oss-20b
```

Adjust model/key values for the actual experiment.

## 5. Install Kathara CLI

The apt repository can fail on fresh GCP VMs when `ppa.kathara.org` is not
resolvable. The project already depends on the Python Kathara package, but that
package does not expose a `kathara` executable by default. Create a small wrapper:

```bash
cd ~/agent-net
source ~/.local/bin/env
uv pip install kathara libtmux

mkdir -p ~/.local/bin
cat > ~/.local/bin/kathara <<'SH'
#!/usr/bin/env bash
exec "$HOME/agent-net/.venv/bin/python" "$HOME/agent-net/.venv/lib/python3.12/site-packages/kathara.py" "$@"
SH

chmod +x ~/.local/bin/kathara
source ~/.local/bin/env
```

Verify Kathara:

```bash
which kathara
kathara --version
kathara check
kathara wipe -f
```

Expected signs:

```text
Current version: 3.8.3
Container run successfully.
```

If a failed Kathara apt source was added earlier, remove it:

```bash
sudo rm -f /etc/apt/sources.list.d/kathara.list
sudo rm -f /usr/share/keyrings/kathara.gpg
sudo apt update
```

## 6. Open Studio On GCP

Start Studio on the VM:

```bash
cd ~/agent-net
source ~/.local/bin/env
uv run nika studio --host 0.0.0.0 --port 8501
```

In GCP Console:

1. Go to **VPC network** -> **Firewall**.
2. Click **Create firewall rule**.
3. Use:

```text
Name: allow-nika-studio-8501
Network: default
Direction of traffic: Ingress
Action on match: Allow
Targets: All instances in the network
Source IPv4 ranges: <YOUR_PUBLIC_IP>/32
Protocols and ports: tcp:8501
```

For quick temporary testing only, `0.0.0.0/0` also works, but it exposes Studio
to the public internet.

Open:

```text
http://<VM_EXTERNAL_IP>:8501
```

## 7. Smoke Tests

Run these before a long benchmark:

```bash
cd ~/agent-net
source ~/.local/bin/env

uv run python -c "import nika; import agent; print('import ok')"
uv run nika --help
docker ps
kathara check
```

Then run a small benchmark from CLI or Studio:

```bash
uv run nika benchmark run --file benchmark/benchmark_test.yaml -a react -b custom -m openai/gpt-oss-20b -n 1
```

The first run can take longer because Docker images are pulled or built.

## 8. Common Failures

`No pyproject.toml found`

Run `uv sync` inside `~/agent-net`, not in `~`.

`Cannot connect to the Docker daemon`

```bash
sudo systemctl start docker
sudo systemctl enable docker
docker ps
```

`permission denied` for Docker

```bash
sudo usermod -aG docker "$USER"
newgrp docker
docker ps
```

`kathara: command not found`

Recreate the wrapper in section 5, then restart Studio so the new process sees
`~/.local/bin` in `PATH`.

`Could not resolve 'ppa.kathara.org'`

Skip apt-based Kathara installation and use the Python-package wrapper in
section 5.

Studio opens but benchmark fails

Inspect the newest result logs:

```bash
cd ~/agent-net
find results -name events.jsonl -printf '%T@ %p\n' | sort -nr | head -5
tail -n 120 <EVENTS_JSONL_PATH>
```

