import ast
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_SECONDS = 10
MEMORY_MB = 512
MAX_FILE_MB = 10
MAX_OUTPUT_CHARS = 1500
MAX_FAILURES_SHOWN = 5


def _cut(text, limit):
    return text if len(text) <= limit else text[: limit - 4] + " ..."


def run_tests_mbpp(
    code, test_list, test_imports=None, timeout=TIMEOUT_SECONDS
):
    """
        Run ``code`` against the assert statements in ``test_list``.

    Returns a JSON string {"success": bool, "output": str}. ``success`` is True
    only if every assertion ran and passed. ``output`` is written for the LLM.
    Never raises.
    """
    try:
        success, output = _run(code, test_list, test_imports, timeout)
    except Exception as exc:  # noqa: BLE001
        success, output = (
            False,
            f"Internal error in run_tests: {type(exc).__name__}: {exc}",
        )
    return json.dumps(
        {"success": success, "output": _cut(output, MAX_OUTPUT_CHARS)}
    )


def _run(code, test_list, test_imports, timeout):
    """
    Validate the MBPP inputs, execute the candidate and its assertions in an
    isolated child process, and return a success flag with a short report.

    The child is started with a separate process group and a resource limit.
    This keeps infinite loops, memory abuse and large output from affecting the
    main agent process.
    """
    if not isinstance(code, str) or not code.strip():
        return False, "No code received: 'code' must be a non-empty string."
    if isinstance(test_list, str):
        test_list = [test_list]
    if not isinstance(test_list, list) or not all(
        isinstance(t, str) for t in test_list
    ):
        return (
            False,
            "'test_list' must be a list of strings (assert statements).",
        )
    tests = [t for t in test_list if t.strip()]
    if not tests:
        return False, "No assertion to run: nothing was verified."
    if isinstance(test_imports, str):
        test_imports = [test_imports]
    imports = "\n".join(str(line) for line in (test_imports or []))
    if not timeout > 0:
        timeout = TIMEOUT_SECONDS

    with tempfile.TemporaryDirectory(prefix="mbpp_") as folder:
        work = Path(folder)
        (work / "in.json").write_text(
            json.dumps({"imports": imports, "code": code, "tests": tests})
        )
        command = [sys.executable, "-I", "-u", "-X", "utf8", __file__, folder]
        timed_out = False
        with open(work / "prints.txt", "wb") as prints:
            process = subprocess.Popen(  # noqa: S603
                command,
                cwd=folder,
                stdin=subprocess.DEVNULL,
                stdout=prints,
                stderr=subprocess.STDOUT,
                env={"PATH": os.environ.get("PATH", os.defpath)},
                start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        printed = (
            (work / "prints.txt")
            .read_bytes()[:2000]
            .decode("utf-8", "replace")
            .strip()
        )
        try:
            report = json.loads((work / "out.json").read_text())
        except (OSError, ValueError):
            report = None

    return _summarize(
        report, tests, printed, process.returncode, timed_out, timeout
    )


def _summarize(report, tests, printed, returncode, timed_out, timeout):
    """
    Turn the child's verdict into (success, short message for the LLM).
    """
    total = len(tests)
    if timed_out:
        message = (
            f"Timeout: the tests did not finish within {timeout:g}s "
            "(infinite loop or too slow)."
        )
    elif report is None:
        message = (
            "The test process stopped before reporting "
            f"(exit code {returncode}). "
            f"The code may call exit() or use more than {MEMORY_MB} MB."
        )
    elif report["load_error"]:
        message = (
            "Your code could not be loaded: "
            f"{report['load_error']}. No assertion was run."
        )
    else:
        failed = [
            (i, r)
            for i, r in enumerate(report["results"], 1)
            if r[0] != "pass"
        ]
        if not failed and len(report["results"]) == total:
            return True, f"All {total} assertions passed."
        lines = [f"{total - len(failed)}/{total} assertions passed. Failures:"]
        for i, (status, detail) in failed[:MAX_FAILURES_SHOWN]:
            verb = "raised" if status == "error" else "failed"
            lines.append(
                f"- assertion {i}/{total} {verb}: "
                f"{_cut(' '.join(tests[i - 1].split()), 120)}"
                + (f" -> {detail}" if detail else "")
            )
        if len(failed) > MAX_FAILURES_SHOWN:
            lines.append(f"- ... and {len(failed) - MAX_FAILURES_SHOWN} more.")
        message = "\n".join(lines)
    if printed:
        message += "\nProgram output:\n" + _cut(printed, 400)
    return False, message


def _limit_resources():
    """
    Best effort: the timeout in the parent remains the main protection.
    """
    try:
        import resource
    except ImportError:
        return
    for limit, megabytes in (
        (resource.RLIMIT_AS, MEMORY_MB),
        (resource.RLIMIT_FSIZE, MAX_FILE_MB),
    ):
        try:
            resource.setrlimit(limit, (megabytes << 20, megabytes << 20))
        except (ValueError, OSError):
            pass


def _describe(exc):
    """
    One line: 'ZeroDivisionError: division by zero (line 2)'.
    """
    if isinstance(exc, SystemExit):
        return "the code called exit()"
    line = exc.lineno if isinstance(exc, SyntaxError) else None
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<solution>":
            line = tb.tb_lineno
        tb = tb.tb_next
    text = (
        f"SyntaxError: {exc.msg}"
        if isinstance(exc, SyntaxError)
        else f"{type(exc).__name__}: {exc}"
    )
    return _cut(text, 200) + (f" (line {line})" if line else "")


def _explain(test, namespace):
    """
    For ``assert A == B`` show the real value of A ('' if unavailable).

    A is evaluated a second time: with side effects (e.g. sort in place),
    the hint can differ from the first run. B is shown as source text.
    """
    try:
        condition = ast.parse(test).body[0].test
        if (
            isinstance(condition, ast.Compare)
            and len(condition.ops) == 1
            and isinstance(condition.ops[0], ast.Eq)
        ):
            got = eval(
                compile(ast.Expression(condition.left), "<test>", "eval"),
                namespace,
            )  # noqa: S307
            return (
                f"got {_cut(repr(got), 80)}, expected "
                f"{_cut(ast.unparse(condition.comparators[0]), 80)}"
            )
    except BaseException:  # noqa: BLE001
        pass
    return ""


def _run_one(test, namespace):
    """
    Execute one assertion in the candidate namespace.

    A passing assertion returns a pass status. An assertion failure includes
    an actual-versus-expected hint when available, while other exceptions are
    reported as the exception type and a concise error message.
    """
    try:
        exec(compile(test, "<test>", "exec"), namespace)  # noqa: S102
        return ["pass", ""]
    except AssertionError:
        return ["fail", _explain(test, namespace)]
    except BaseException as exc:  # noqa: BLE001
        return ["error", _describe(exc)]


def _child(folder):
    """
    Load and run a candidate inside the isolated child process.

    The candidate, imports and assertions are read from the temporary folder.
    Results are written to `out.json`, containing loading errors or the status
    of every individual assertion.
    """
    data = json.loads((folder / "in.json").read_text())
    _limit_resources()
    namespace = {"__name__": "candidate_solution"}
    report = {"load_error": None, "results": []}
    try:
        exec(
            compile(data["imports"], "<imports>", "exec"), namespace
        )  # noqa: S102
        exec(
            compile(data["code"], "<solution>", "exec"), namespace
        )  # noqa: S102
    except BaseException as exc:  # noqa: BLE001
        report["load_error"] = _describe(exc)
    else:
        report["results"] = [
            _run_one(test, namespace) for test in data["tests"]
        ]
    (folder / "out.json").write_text(json.dumps(report))


if __name__ == "__main__":
    _child(Path(sys.argv[1]))
