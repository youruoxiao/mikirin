"""Small, offline orchestration tests for scripts/build_p1.py.

These tests deliberately mock every database/model boundary.  They verify the
stage order and safety gates without opening a real database or making a
network request.
"""

from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_p1.py"
SPEC = importlib.util.spec_from_file_location("mikirin_build_p1_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_p1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_p1
SPEC.loader.exec_module(build_p1)


def _context(*, stages: dict[str, dict[str, object]] | None = None) -> SimpleNamespace:
    stage_state = stages or {
        name: {"status": "pending", "attempts": 0}
        for name in build_p1.STAGE_NAMES
    }
    progress = {
        "config": {
            "slang_arbitration_limit": 5_000,
            "llm_workers": 64,
            "implicit_pair_limit": 500,
            "implicit_evidence_per_pair": 3,
        },
        "stages": stage_state,
        "user_acceptance": {},
    }
    context = SimpleNamespace(
        root=ROOT,
        build_id="build-1",
        run_id="run-1",
        model_contract={
            "provider": "siliconflow",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "Qwen/Qwen3-8B",
            "key_env": "SILICONFLOW_API_KEY",
        },
        slang_config=SimpleNamespace(max_subjects=None),
        p1_config=object(),
        progress=progress,
        saves=0,
    )

    def save() -> None:
        context.saves += 1

    context.save = save
    return context


class BuildP1ScriptTests(unittest.TestCase):
    def _args(self, directory, *options):
        return build_p1.build_parser().parse_args([
            "prepare", "--root", str(directory), "--version", "fixture-v2",
            "--build-id", "fixture-build", "--run-id", "new-run", *options,
        ])

    def _neighbors(self, directory):
        path = Path(directory) / "neighbors.json"
        path.write_text('{"锯":[{"entity_id":"subject:chainsaw","score":0.97}]}', encoding="utf-8")
        return path

    def test_cli_freezes_inputs_and_resume_restores_them_without_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._neighbors(directory)
            args = self._args(directory, "--incremental-base-run-id", "old-run",
                              "--vector-neighbors-file", str(source))
            # Execute the actual parser, context and progress stage boundary;
            # only database preparation is replaced by an offline fake.
            with patch.object(build_p1, "_prepare", return_value={"fixture": True}):
                code, _ = build_p1.run(args)
            self.assertEqual(code, 0)
            context = args._p1_context
            saved = json.loads(context.progress_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["config"]["slang"]["incremental_base_run_id"], "old-run")
            self.assertEqual(saved["config"]["slang"]["vector_neighbors_by_candidate"]["锯"][0]["score"], .97)
            self.assertEqual(saved["config"]["slang_neighbors_input"]["sha256"],
                             "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest())
            resumed = build_p1._resolve_context(self._args(directory))
            self.assertEqual(resumed.slang_config, context.slang_config)
            with patch.object(build_p1, "_require_prepared", return_value=SimpleNamespace(target=Path(directory) / "fixture.db")), \
                 patch.object(build_p1, "discover_slang", return_value={}) as discovery, \
                 patch.object(build_p1, "load_arbitration_candidates", return_value=[]), \
                 patch.object(build_p1, "StructuredAPIClient") as client:
                build_p1._run_slang(resumed)
            self.assertEqual(discovery.call_args.kwargs["config"], context.slang_config)
            client.assert_not_called()

    def test_resume_rejects_changed_neighbors_even_when_file_flag_is_omitted(self):
        for change in ("score", "whitespace", "missing"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                source = self._neighbors(directory)
                ctx = build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(source)))
                progress = ctx.progress_path.read_bytes()
                if change == "missing":
                    source.unlink()
                    error = FileNotFoundError
                else:
                    text = source.read_text(encoding="utf-8")
                    source.write_text(text.replace("0.97", "0.96") if change == "score" else text + "\n", encoding="utf-8")
                    error = ValueError
                with self.assertRaises(error):
                    build_p1._resolve_context(self._args(directory))
                self.assertEqual(ctx.progress_path.read_bytes(), progress)

    def test_resume_checks_frozen_payload_and_supports_identical_file_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._neighbors(directory)
            ctx = build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(source)))
            relocated = Path(directory) / "same-bytes.json"
            relocated.write_bytes(source.read_bytes())
            source.unlink()
            resumed = build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(relocated)))
            self.assertEqual(resumed.slang_config, ctx.slang_config)
            saved = json.loads(ctx.progress_path.read_text(encoding="utf-8"))
            saved["config"]["slang"]["vector_neighbors_by_candidate"]["锯"][0]["score"] = .1
            ctx.progress_path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(relocated)))

    def test_resume_cannot_change_base_or_add_neighbors_to_existing_build(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = build_p1._resolve_context(self._args(directory, "--incremental-base-run-id", "old-run"))
            before = ctx.progress_path.read_bytes()
            source = self._neighbors(directory)
            with self.assertRaisesRegex(ValueError, "incremental-base-run-id"):
                build_p1._resolve_context(self._args(directory, "--incremental-base-run-id", "different"))
            with self.assertRaisesRegex(ValueError, "no frozen neighbors"):
                build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(source)))
            self.assertEqual(ctx.progress_path.read_bytes(), before)

    def test_new_run_cannot_use_itself_as_incremental_base(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must differ"):
                build_p1._resolve_context(self._args(directory, "--incremental-base-run-id", "new-run"))
            self.assertFalse((Path(directory) / "data").exists())

    def test_invalid_neighbors_are_rejected_before_progress_creation(self):
        inputs = ['[]', '{"锯":{}}', '{"锯":[{"entity_id":"x","score":NaN}]}',
                  '{"锯":[{"entity_id":"x","score":true}]}',
                  '{"锯":[{"entity_id":"x","score":1.1}]}',
                  '{"锯":[{"score":0.9}]}', '{"锯":[],"锯":[]}',
                  '{"ABC":[],"abc":[]}', '{"锯":[{"entity_id":"x","score":0.9,"evidence":[]}]}']
        for content in inputs:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "invalid.json"
                source.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_p1._resolve_context(self._args(directory, "--vector-neighbors-file", str(source)))
                self.assertFalse((Path(directory) / "data").exists())

    def test_new_flags_work_before_or_after_subcommand(self):
        for argv in (["--incremental-base-run-id", "base", "--vector-neighbors-file", "n.json", "slang"],
                     ["slang", "--slang-incremental-base-run-id", "base", "--slang-vector-neighbors-file", "n.json"]):
            args = build_p1.build_parser().parse_args(argv)
            self.assertEqual(args.incremental_base_run_id, "base")
            self.assertEqual(args.vector_neighbors_file, "n.json")

    def test_old_completed_graph_stage_is_rebuilt_once_for_heterogeneous_version(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            args.stage = "graph"
            ctx = build_p1._resolve_context(args)
            for name in ("prepare", "graph", "pagerank"):
                ctx.progress["stages"][name]["status"] = "completed"
            ctx.save()
            def graph(context):
                context.progress["stages"]["pagerank"].update(
                    status="completed", result={"task_graph_version": build_p1.TASK_GRAPH_VERSION})
                return {"task_graph_version": build_p1.TASK_GRAPH_VERSION}
            with patch.object(build_p1, "_run_graph", side_effect=graph) as operation:
                build_p1.run(args)
                build_p1.run(args)
            self.assertEqual(operation.call_count, 1)

    def test_default_arbitration_contract_is_siliconflow_qwen_8b_not_embedding_model(self) -> None:
        args = SimpleNamespace(
            qwen_model=None,
            siliconflow_base_url=None,
            siliconflow_key_env=None,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            contract = build_p1._read_runtime_embedding_contract(Path(directory), args)
        self.assertEqual(contract["provider"], "siliconflow")
        self.assertEqual(contract["model"], "Qwen/Qwen3-8B")

    def test_commands_start_without_work_data_or_installed_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory) / "checkout"
            for name in ("scripts", "src"):
                shutil.copytree(
                    ROOT / name, checkout / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
                )
            self.assertFalse((checkout / "work").exists())
            for script in (
                "build_p1", "prepare_p1_manual_review", "extract_discussion_maps",
                "check_theme_quality",
            ):
                # -E/-S exclude PYTHONPATH and site-packages/editable installs.
                commands = (
                    ([str(checkout / "scripts" / f"{script}.py")], directory),
                    (["-m", f"scripts.{script}"], checkout),
                )
                for command, cwd in commands:
                    with self.subTest(script=script, command=command):
                        result = subprocess.run(
                            [sys.executable, "-B", "-E", "-S", "-X", "utf8", *command, "--help"],
                            cwd=cwd, capture_output=True, text=True, encoding="utf-8", timeout=30,
                        )
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn("usage:", result.stdout)
            self.assertFalse((checkout / "data").exists())
            self.assertFalse((checkout / "db").exists())
            self.assertFalse((checkout / "work").exists())

    def test_slang_is_discovery_then_qwen_arbitration_then_apply_and_loads_dotenv(self) -> None:
        context = _context()
        paths = SimpleNamespace(target=ROOT / "db" / "side.db")
        order: list[str] = []
        profile = SimpleNamespace(
            key=lambda: (order.append("key") or "secret"),
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen3-8B",
            provider="siliconflow",
        )
        arbitrator_kwargs: dict[str, object] = {}

        class FakeArbitrator:
            def __init__(self, **kwargs: object) -> None:
                order.append("arbitrator")
                arbitrator_kwargs.update(kwargs)

            def arbitrate_batches(self, candidates: object) -> dict[str, object]:
                order.append("arbitrate")
                return {"decisions": [{"id": "decision-1"}]}

        with (
            patch.object(build_p1, "_require_prepared", return_value=paths),
            patch.object(build_p1, "discover_slang", side_effect=lambda *a, **k: order.append("discover") or {"subjects": 2}),
            patch.object(build_p1, "load_arbitration_candidates", side_effect=lambda *a, **k: order.append("candidates") or [{"id": "candidate-1"}]),
            patch.object(build_p1, "load_env_file", side_effect=lambda path: order.append(f"env:{path.name}")),
            patch.object(build_p1, "ModelProfile", return_value=profile),
            patch.object(build_p1, "StructuredAPIClient", side_effect=lambda **kwargs: order.append("client") or Mock()),
            patch.object(build_p1, "SlangArbitrator", FakeArbitrator),
            patch.object(build_p1, "apply_arbitration_results", side_effect=lambda *a, **k: order.append("apply") or {"confirmed": 1}),
        ):
            result = build_p1._run_slang(context)

        self.assertEqual(order, ["discover", "candidates", "env:.env", "key", "client", "arbitrator", "arbitrate", "apply"])
        self.assertEqual(result["model"]["model"], "Qwen/Qwen3-8B")
        self.assertEqual(result["model"]["provider"], "siliconflow")
        self.assertEqual(result["arbitrated"], 1)
        self.assertEqual(arbitrator_kwargs["workers"], 64)

    def test_graph_materializes_derived_global_and_task_pagerank_as_one_stage(self) -> None:
        context = _context()
        paths = SimpleNamespace(target=ROOT / "db" / "side.db")
        order: list[str] = []

        with (
            patch.object(build_p1, "_require_prepared", return_value=paths),
            patch.object(build_p1, "materialize_derived_edges", side_effect=lambda *a, **k: order.append("derived") or {"same_season": 1}),
            patch.object(build_p1, "deterministic_edge_digest", side_effect=lambda *a, **k: order.append("digest") or "edge-digest"),
            patch.object(build_p1, "materialize_node_pagerank", side_effect=lambda *a, **k: order.append("global") or {"same_season": 2}),
            patch.object(build_p1, "materialize_task_pagerank", side_effect=lambda *a, **k: order.append("task") or {"similar_work": 3}),
        ):
            result = build_p1._run_graph(context)

        self.assertEqual(order, ["derived", "digest", "global", "task"])
        self.assertEqual(result["derived"], {"same_season": 1})
        self.assertEqual(result["global"], {"same_season": 2})
        self.assertEqual(result["task_personalized"], {"similar_work": 3})
        self.assertEqual(context.progress["stages"]["pagerank"]["status"], "completed")

    def test_implicit_runs_pass_a_then_qwen_then_apply_and_loads_dotenv(self) -> None:
        context = _context(stages={
            name: {"status": "completed", "attempts": 1}
            for name in build_p1.STAGE_NAMES
        })
        paths = SimpleNamespace(target=ROOT / "db" / "side.db")
        order: list[str] = []
        profile = SimpleNamespace(
            key=lambda: (order.append("key") or "secret"),
            base_url="https://api.siliconflow.cn/v1",
            model="Qwen/Qwen3-8B",
            provider="siliconflow",
        )
        arbitrator_kwargs: dict[str, object] = {}

        class FakeArbitrator:
            def __init__(self, **kwargs: object) -> None:
                order.append("arbitrator")
                arbitrator_kwargs.update(kwargs)

            def arbitrate_many(self, candidates: object) -> list[dict[str, str]]:
                order.append("qwen")
                return [{"id": "decision-1"}]

        with (
            patch.object(build_p1, "_require_prepared", return_value=paths),
            patch.object(build_p1, "load_implicit_candidates", side_effect=lambda *a, **k: order.append("pass-a") or [{"id": "candidate-1"}]),
            patch.object(build_p1, "load_env_file", side_effect=lambda path: order.append(f"env:{path.name}")),
            patch.object(build_p1, "ModelProfile", return_value=profile),
            patch.object(build_p1, "StructuredAPIClient", side_effect=lambda **kwargs: order.append("client") or Mock()),
            patch.object(build_p1, "ImplicitRelationArbitrator", FakeArbitrator),
            patch.object(build_p1, "apply_implicit_results", side_effect=lambda *a, **k: order.append("apply") or {"implicit_edges": 1}),
        ):
            result = build_p1._run_implicit(context)

        self.assertEqual(order, ["pass-a", "env:.env", "key", "client", "arbitrator", "qwen", "apply"])
        self.assertEqual(result["decisions"], 1)
        self.assertEqual(arbitrator_kwargs["workers"], 64)

    def test_publish_runs_release_gate_before_publisher(self) -> None:
        context = _context(stages={
            name: {"status": "completed", "attempts": 1}
            for name in build_p1.STAGE_NAMES
        })
        context.progress["stages"]["validate"]["result"] = {"publishable": True}
        paths = SimpleNamespace(target=ROOT / "db" / "side.db")
        publisher = Mock()

        with (
            patch.object(build_p1, "_require_prepared", return_value=paths),
            patch.object(build_p1, "p1_release_issues", return_value=["missing user acceptance"]),
            patch.object(build_p1, "publish_side_build", publisher),
        ):
            with self.assertRaisesRegex(RuntimeError, "release gate failed"):
                build_p1._run_publish(context, requested=True)

        publisher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
