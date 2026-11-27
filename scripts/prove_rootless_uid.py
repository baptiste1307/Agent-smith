"""Minimal proof of the moulinette rootless-Docker bug (no agent, no API key).

It calls the exact function the moulinette uses to send the patch into the
SWE-bench container (swebench.harness.docker_utils.copy_to_container), in a
throwaway container, and compares it with the same copy done with uid 0.

Usage (from the repository root):

    cd moulinette
    uv run python ../scripts/prove_rootless_uid.py [image]

`image` defaults to python:3.11-slim (any local image with /bin/sh works).
"""

import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

import docker
from swebench.harness.docker_utils import copy_to_container

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "python:3.11-slim"


def subuid_range(user: str) -> str:
    try:
        for line in Path("/etc/subuid").read_text().splitlines():
            if line.startswith(f"{user}:"):
                return line
    except OSError:
        pass
    return "(not found)"


def main() -> None:
    client = docker.from_env()
    security = client.info().get("SecurityOptions", [])
    uid, gid = os.getuid(), os.getgid()
    print("== Environment")
    print(f"host uid/gid           : {uid}/{gid}")
    user = os.environ.get("USER", "")
    print(f"/etc/subuid            : {subuid_range(user)}")
    print(f"docker rootless        : {any('rootless' in s for s in security)}")
    print("inside a rootless container only uids 0..65535 exist")

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "patch.diff"
        src.write_text("diff --git a/x b/x\n")

        # What swebench's copy_to_container puts in the tar header
        with tarfile.open(fileobj=io.BytesIO(), mode="w") as tar:
            info = tar.gettarinfo(str(src), arcname="patch.diff")
        print(f"tar header uid/gid     : {info.uid}/{info.gid}  "
              "(tar.add keeps the host owner)")

        container = client.containers.run(
            IMAGE, "sleep 120", detach=True, remove=True
        )
        try:
            print("\n== 1. swebench copy_to_container (moulinette code path)")
            try:
                copy_to_container(container, src, Path("/tmp/patch.diff"))
                print("OK (no bug on this machine)")
            except Exception as e:
                print(f"FAILED: {e}")

            print("\n== 2. same file, tar header forced to uid/gid 0")
            data = src.read_bytes()
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                info = tarfile.TarInfo("patch.diff")
                info.size, info.uid, info.gid = len(data), 0, 0
                tar.addfile(info, io.BytesIO(data))
            container.put_archive("/tmp", buf.getvalue())
            out = container.exec_run("ls -ln /tmp/patch.diff").output
            print(f"OK: {out.decode().strip()}")
        finally:
            container.kill()


if __name__ == "__main__":
    main()
