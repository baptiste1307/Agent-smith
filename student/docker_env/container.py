"""Docker container management and lifecycle for SWE-bench execution.

Implements the Docker execution environment required by Section V.4:
- Pulling and starting container instances for SWE-bench tasks
- Safely executing commands and evaluation scripts inside the container
- File synchronization between host and container
- Guaranteed container cleanup via atexit and signal handlers (SIGTERM, SIGINT)
"""

import atexit
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


class DockerContainer:
    """Manages the lifecycle of a Docker container for a SWE-bench task."""

    _active_containers: set[str] = set()
    _cleanup_registered: bool = False

    def __init__(
        self,
        image_name: str,
        container_name: str | None = None,
        auto_cleanup: bool = True,
    ) -> None:
        """Initialize the Docker container manager.

        Args:
            image_name: Docker image name on DockerHub
                (e.g. swebench/sweb.eval.x86_64.sympy_1776_sympy-23534:latest).
            container_name: Optional explicit name for the container.
            auto_cleanup: If True, registers process exit hooks to delete
                the container.
        """
        self.image_name = image_name
        safe_name = re.sub(
            r"[^a-zA-Z0-9_.-]",
            "_",
            container_name or f"agent_swe_{int(time.time())}",
        )
        self.container_name = safe_name
        self.container_id: str | None = None
        self.auto_cleanup = auto_cleanup
        self._is_cleaned_up = False

        if auto_cleanup:
            self._register_global_cleanup()

    @classmethod
    def _register_global_cleanup(cls) -> None:
        """Register signal and exit handlers once."""
        if cls._cleanup_registered:
            return

        def _cleanup_all() -> None:
            for cid in list(cls._active_containers):
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", cid],
                        check=False,
                        capture_output=True,
                        timeout=15,
                    )
                except Exception:
                    pass
            cls._active_containers.clear()

        atexit.register(_cleanup_all)

        def _signal_handler(signum: int, frame: Any) -> None:
            _cleanup_all()
            # Re-raise or exit cleanly
            sys.exit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _signal_handler)
            except (ValueError, OSError):
                # May fail if not in main thread
                pass

        cls._cleanup_registered = True

    @staticmethod
    def is_docker_available() -> bool:
        """Check if Docker daemon is accessible and responding."""
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def is_image_present(self) -> bool:
        """Check if the target image exists locally."""
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", self.image_name],
                capture_output=True,
                timeout=10,
            )
            return res.returncode == 0
        except Exception:
            return False

    def pull_image(self, timeout: float = 600.0) -> bool:
        """Pull the Docker image from DockerHub if not present."""
        if self.is_image_present():
            return True

        pull_cmd = ["docker", "pull"]
        if "x86_64" in self.image_name or sys.platform == "darwin":
            pull_cmd.extend(["--platform", "linux/amd64"])
        pull_cmd.append(self.image_name)

        try:
            res = subprocess.run(
                pull_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.returncode == 0
        except Exception:
            return False

    def start(self, pull_if_missing: bool = True) -> str:
        """Start a detached container running an idle command.

        Returns:
            The started container ID.

        Raises:
            RuntimeError: If Docker is unavailable or start fails.
        """
        if not self.is_docker_available():
            raise RuntimeError(
                "Docker daemon is not running or not accessible."
            )

        if pull_if_missing and not self.is_image_present():
            pulled = self.pull_image()
            if not pulled:
                raise RuntimeError(
                    f"Failed to pull Docker image: {self.image_name}"
                )

        cmd = [
            "docker",
            "run",
            "-d",
        ]
        if "x86_64" in self.image_name or sys.platform == "darwin":
            cmd.extend(["--platform", "linux/amd64"])
        cmd.extend(
            [
                "--name",
                self.container_name,
                self.image_name,
                "tail",
                "-f",
                "/dev/null",
            ]
        )

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode != 0:
            raise RuntimeError(
                f"Failed to start container '{self.container_name}': "
                f"{res.stderr}"
            )

        cid = res.stdout.strip()
        self.container_id = cid
        self.__class__._active_containers.add(cid)
        return cid

    def exec_run(
        self,
        command: str | list[str],
        workdir: str = "/testbed",
        user: str | None = None,
        timeout: float = 300.0,
    ) -> tuple[int, str, str]:
        """Execute a shell command inside the running container.

        Args:
            command: Shell command string or argument list.
            workdir: Working directory inside container (default: /testbed).
            user: Optional username inside container.
            timeout: Maximum execution time in seconds.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        if not self.container_id:
            raise RuntimeError("Container is not running.")

        cmd = ["docker", "exec", "-w", workdir]
        if user:
            cmd.extend(["-u", user])

        cmd.append(self.container_id)

        if isinstance(command, str):
            cmd.extend(["/bin/bash", "-c", command])
        else:
            cmd.extend(command)

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as e:
            stdout = (
                e.stdout
                if isinstance(e.stdout, str)
                else (e.stdout.decode() if e.stdout else "")
            )
            stderr = (
                e.stderr
                if isinstance(e.stderr, str)
                else (e.stderr.decode() if e.stderr else "")
            )
            return (
                124,
                stdout,
                f"{stderr}\nCommand timed out after {timeout} seconds.",
            )

    def copy_from_container(
        self, container_path: str, host_path: str | Path
    ) -> bool:
        """Copy files or directories from the container to the host."""
        if not self.container_id:
            raise RuntimeError("Container is not running.")

        dest = str(Path(host_path).resolve())
        cmd = ["docker", "cp", f"{self.container_id}:{container_path}", dest]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        return res.returncode == 0

    def copy_to_container(
        self, host_path: str | Path, container_path: str
    ) -> bool:
        """Copy a file or directory from host into the container.

        For regular files, streams bytes via `docker exec -i` to avoid
        rootless Docker `lchown` errors when host UID exceeds subuid range.
        """
        if not self.container_id:
            raise RuntimeError("Container is not running.")

        src_path = Path(host_path).resolve()
        if src_path.is_file():
            data = src_path.read_bytes()
            res = subprocess.run(
                [
                    "docker",
                    "exec",
                    "-i",
                    self.container_id,
                    "/bin/sh",
                    "-c",
                    'cat > "$1"',
                    "_",
                    container_path,
                ],
                input=data,
                capture_output=True,
                timeout=60,
            )
            return res.returncode == 0

        cmd = [
            "docker",
            "cp",
            str(src_path),
            f"{self.container_id}:{container_path}",
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        return res.returncode == 0

    def apply_patch(self, patch: str) -> bool:
        """Apply a unified git diff patch inside the container's repository."""
        if not patch or not patch.strip():
            return False

        with tempfile.NamedTemporaryFile(
            "w", suffix=".diff", delete=False
        ) as f:
            f.write(patch)
            temp_patch = f.name

        try:
            copied = self.copy_to_container(temp_patch, "/tmp/patch.diff")
            if not copied:
                return False

            # Try git apply variants
            for git_cmd in (
                "git apply --verbose /tmp/patch.diff",
                "git apply --verbose --reject /tmp/patch.diff",
                "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
            ):
                code, _, _ = self.exec_run(git_cmd, workdir="/testbed")
                if code == 0:
                    return True
            return False
        finally:
            if os.path.exists(temp_patch):
                os.remove(temp_patch)

    def run_eval_script(
        self, script_content: str, timeout: float = 300.0
    ) -> tuple[int, str]:
        """Execute an eval script inside the container and return output."""
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            self.copy_to_container(temp_script, "/eval.sh")
            self.exec_run("chmod +x /eval.sh", workdir="/testbed")
            code, stdout, stderr = self.exec_run(
                "/bin/bash /eval.sh",
                workdir="/testbed",
                timeout=timeout,
            )
            combined = f"{stdout}\n{stderr}".strip()
            return code, combined
        finally:
            if os.path.exists(temp_script):
                os.remove(temp_script)

    def stop(self) -> None:
        """Stop and remove the container."""
        if self._is_cleaned_up or not self.container_id:
            return

        cid = self.container_id
        try:
            subprocess.run(
                ["docker", "rm", "-f", cid],
                check=False,
                capture_output=True,
                timeout=15,
            )
        except Exception:
            pass
        finally:
            self.__class__._active_containers.discard(cid)
            self._is_cleaned_up = True
            self.container_id = None

    def __enter__(self) -> "DockerContainer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
