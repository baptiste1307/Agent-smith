"""Run `moulinette_eval` with a rootless-Docker-safe copy_to_container.

Why: the moulinette copies the patch into the SWE-bench container with
`swebench.harness.docker_utils.copy_to_container`, which builds a tar that
keeps the host UID of the file (e.g. 102114 on 42 machines). Rootless Docker
only maps UIDs 0-65535 inside the container, so `lchown` fails:

    failed to Lchown "/tmp/patch.diff" for UID 102114 ...: invalid argument

and every SWE-bench validation is reported FAILED before the tests even run.

This script replaces that function in memory only (uid/gid 0 in the tar
header), then runs the normal moulinette CLI. No moulinette file is changed.
It is NOT used by exams/exam_swebench.sh: it is a tool to re-check a
solution.json by hand.

Usage (from the repository root):

    cd moulinette
    uv run python ../scripts/validate_rootless.py validate swebench \
        ../evaluations/swebench/<date>/<task_id>/task.json \
        ../evaluations/swebench/<date>/<task_id>/solution.json

Any other `moulinette_eval` subcommand works the same way.
"""

import io
import os
import sys
import tarfile
from pathlib import Path

import moulinette.swebench.interact as interact
from moulinette.__main__ import main


def copy_to_container(container, src: Path, dst: Path) -> None:
    """Copy a host file into the container, owned by root inside it."""
    data = Path(src).read_bytes()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name=dst.name)
        info.size = len(data)
        info.mode = 0o644
        info.uid = info.gid = 0
        tar.addfile(info, io.BytesIO(data))
    container.exec_run(f"mkdir -p {dst.parent}")
    container.put_archive(os.path.dirname(dst), buf.getvalue())


if __name__ == "__main__":
    interact.copy_to_container = copy_to_container
    sys.argv = ["moulinette_eval", *sys.argv[1:]]
    main()
