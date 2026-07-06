"""Containerlab min 5-stage CLOS fabric (clos02)."""

from __future__ import annotations

import shutil
import subprocess

from nika.net_env.containerlab.base import ContainerlabNetworkEnv
from nika.runtime.containerlab import render_topology


class ContainerlabMin5Clos(ContainerlabNetworkEnv):
    # ref: https://containerlab.dev/lab-examples/min-5clos/#description
    LAB_NAME = "min5clos"
    TOPO_LEVEL = "medium"
    TOPO_SIZE = 14
    TAGS = ["clos", "srl", "bgp", "containerlab", "fabric"]
    DESC = "5-stage CLOS fabric with Nokia SR Linux (Containerlab min5clos / clos02)."

    def _prepare_runtime_files(self) -> None:
        super()._prepare_runtime_files()
        lab_name = self.name
        if not lab_name or self.runtime_workdir is None:
            raise ValueError("Lab name is required before deploy.")

        configs_src = self.lab_dir / "configs"
        configs_dst = self.runtime_workdir / "configs"
        if configs_dst.exists():
            shutil.rmtree(configs_dst)
        shutil.copytree(configs_src, configs_dst)

        setup_template = self.lab_dir / "setup.sh.tmpl"
        setup_dst = self.runtime_workdir / "setup.sh"
        render_topology(setup_template, lab_name=lab_name, output_path=setup_dst)
        setup_dst.chmod(0o755)

    def deploy(self) -> None:
        already_existed = self.lab_exists()
        super().deploy()
        if already_existed:
            return
        self._run_setup()

    def _run_setup(self) -> None:
        self._ensure_runtime_files()
        if self.runtime_workdir is None:
            raise ValueError("runtime_workdir is required for setup.")
        setup_script = self.runtime_workdir / "setup.sh"
        if not setup_script.is_file():
            raise FileNotFoundError(f"Missing setup script: {setup_script}")
        result = subprocess.run(
            ["bash", str(setup_script)],
            cwd=str(self.runtime_workdir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"min5clos setup.sh failed: {result.stderr or result.stdout}"
            )

    def verify_lab(self) -> dict:
        from nika.net_env.containerlab.min5clos.verify import verify_min5clos_lab

        return verify_min5clos_lab(self._build_runtime(), scenario_name=self.LAB_NAME)
