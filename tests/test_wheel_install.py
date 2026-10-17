"""Offline wheel smoke test: python tests/test_wheel_install.py <wheel path>."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


CLI_SMOKE = r'''
import runpy
import sys

network_attempts = []

def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        network_attempts.append(event)
        raise RuntimeError("Network access is forbidden in CLI smoke tests")

sys.addaudithook(deny_network)
sys.argv = ["mikirin", *sys.argv[1:]]
try:
    runpy.run_module("mikirin", run_name="__main__")
finally:
    assert not network_attempts, network_attempts
'''


SMOKE = r'''
import sqlite3
import runpy
import sys
from contextlib import closing
from pathlib import Path

def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("Network access is forbidden in wheel smoke tests")

sys.addaudithook(deny_network)

import mikirin
from mikirin.cli import main
from mikirin.layout import ProjectLayout, discover_root
from mikirin.prompts import PromptBuilder
from mikirin.publish import Publisher
from mikirin.retrieval_evaluation import load_benchmark
from mikirin.p1_evaluation import load_graph_queries

assert Path(mikirin.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert len(load_benchmark()["questions"]) == 60
assert len(load_graph_queries()) == 30
root = Path.cwd().resolve()
assert discover_root() == root
assert main(["init"]) == 0
layout = ProjectLayout(root)
assert layout.pipeline_db.is_file()
database = Publisher(layout).current()
assert database is not None
with closing(sqlite3.connect(database)) as connection:
    assert connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0] == 3
    assert connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 9
builder = PromptBuilder(layout)
defaults = builder.build().stable
assert "Mikirin" in defaults and "search_anime" in defaults
assert not builder.soul_path.exists() and not builder.tools_path.exists()
builder.soul_path.write_text("wheel user override", encoding="utf-8")
builder.user_path.write_text("existing user profile", encoding="utf-8")
builder.memory_path.write_text("existing user memory", encoding="utf-8")
assert builder.build().stable.startswith("wheel user override\n\n")
assert "search_anime" in builder.build().stable
assert builder.user_path.read_text(encoding="utf-8") == "existing user profile"
assert builder.memory_path.read_text(encoding="utf-8") == "existing user memory"
assert main(["status"]) == 0

# Run the same regression cases against the installed distribution, with -I
# preventing the source checkout or PYTHONPATH from satisfying imports.
test_path = sys.argv[1]
sys.argv = [test_path, "-v"]
runpy.run_path(test_path, run_name="__main__")
'''


def smoke_test(wheel: Path) -> None:
    wheel = wheel.resolve(strict=True)
    regression_tests = Path(__file__).with_name("test_layout.py").resolve()
    with tempfile.TemporaryDirectory(prefix="mikirin-wheel-") as directory:
        temporary = Path(directory)
        environment = temporary / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        work = temporary / "unrelated project" / "任意工作目录"
        work.mkdir(parents=True)
        (work.parent / "pyproject.toml").write_text(
            '[project]\nname = "unrelated"\n', encoding="utf-8"
        )
        subprocess.run(
            [str(python), "-I", "-m", "pip", "--isolated", "install", "--no-index",
             "--no-deps", "--no-cache-dir", "--disable-pip-version-check", str(wheel)],
            cwd=work,
            check=True,
        )

        def run_cli(*arguments: str, input_text: str = "") -> str:
            result = subprocess.run(
                [str(python), "-I", "-X", "utf8", "-c", CLI_SMOKE, *arguments],
                cwd=work,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
            )
            if result.returncode:
                raise AssertionError(f"CLI {arguments}:\n{result.stdout}\n{result.stderr}")
            return result.stdout

        for prefix in ((), ("--root", str(temporary / "explicit root"))):
            run_cli(*prefix, "demo")
            hits = json.loads(run_cli(*prefix, "search", "放映部"))
            assert any(hit["canonical_name"] == "屋顶放映部" for hit in hits), hits
            for mode in ("fts", "anchored", "evidence", "hybrid"):
                result = json.loads(run_cli(*prefix, "research-search", "放映部", "--mode", mode))
                assert result["entities"], result
            json.loads(run_cli(*prefix, "rss", "status"))
            json.loads(run_cli(*prefix, "memory-queue", "status"))
            # Exercise ordinary input, slash-command search, and all offline exits.
            for ending in ("/quit\n", "/exit\n", ""):
                output = run_cli(
                    *prefix, "tui", "--browse-only",
                    input_text="放映部\n/search 末班猫\n" + ending,
                )
                assert "屋顶放映部" in output and "末班电车与猫" in output, output
                assert "error>" not in output and "API key" not in output, output
                assert "退出前结算" not in output, output
            print(f"Offline demo/search/TUI smoke passed: {prefix or 'cwd'}", flush=True)

        subprocess.run(
            [str(python), "-I", "-X", "utf8", "-c", SMOKE, str(regression_tests)],
            cwd=work,
            check=True,
            timeout=60,
        )
        # Also exercise the generated console entry point without relying on PATH.
        executable = scripts / ("mikirin.exe" if os.name == "nt" else "mikirin")
        subprocess.run(
            [str(executable), "--help"], cwd=work, check=True,
            stdout=subprocess.PIPE, timeout=30,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tests/test_wheel_install.py <wheel path>")
    smoke_test(Path(sys.argv[1]))
