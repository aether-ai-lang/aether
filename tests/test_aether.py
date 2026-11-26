"""End-to-end tests for the Aether v0.2 compiler, checker, and runners."""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(ROOT))


def run_cli(*args: str, cwd: pathlib.Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(ROOT / "aether.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def write_temp(source: str, name: str = "case.ae") -> pathlib.Path:
    directory = pathlib.Path(tempfile.mkdtemp(prefix="aether_test_"))
    path = directory / name
    path.write_text(source, encoding="utf-8", newline="\n")
    return path


def native_toolchain():
    from aetherc.cbackend import find_toolchain

    return find_toolchain()


class AetherCliTests(unittest.TestCase):
    def test_check_hello_is_ok(self) -> None:
        result = run_cli("check", "hello_aether.ae")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["summary"]["tasks"], 1)
        self.assertEqual(payload["diagnostics"], [])

    def test_run_streams_and_reports_metrics(self) -> None:
        result = run_cli("run", "hello_aether.ae", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b", result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["task"], "HelloAether")
        chunks = [event for event in payload["events"] if event["event"] == "chunk"]
        self.assertEqual(len(chunks), 8)
        self.assertGreater(payload["result"]["gflops"], 1.0)
        self.assertGreater(payload["result"]["gflops_per_s"], 1.0)
        self.assertTrue(payload["result"]["digest"].startswith("0x"))
        self.assertTrue(pathlib.Path(payload["compile"]["artifact"]).is_file())

    def test_generated_plan_is_real_code(self) -> None:
        result = run_cli("run", "hello_aether.ae", "--emit-only", "--target", "python")
        self.assertEqual(result.returncode, 0)
        compiled = (ROOT / ".aether_build" / "hello_aether.py").read_text(encoding="utf-8")
        self.assertIn("def build(runtime):", compiled)
        self.assertIn("rt.tensor(2048, 1024)", compiled)
        self.assertIn("rt.neural_net(", compiled)
        self.assertIn("rt.stream(net, data)", compiled)
        self.assertIn("'speed': 'max'", compiled)
        self.assertIn("PLAN =", compiled)

    def test_native_codegen_is_standalone_c(self) -> None:
        if native_toolchain() is None:
            self.skipTest("no C compiler available")
        result = run_cli(
            "run", "hello_aether.ae", "--emit-only", "--target", "c", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "c")
        source = pathlib.Path(payload["artifact"]).read_text(encoding="utf-8")
        self.assertIn("#include <stdio.h>", source)
        self.assertIn("int main(", source)
        self.assertIn("task_HelloAether(void)", source)
        self.assertIn("ae_stream(&net, &data,", source)
        self.assertIn("task_done", source)

    def test_native_run_is_default_when_compiler_present(self) -> None:
        payload = json.loads(run_cli("run", "hello_aether.ae", "--json").stdout)
        has_compiler = native_toolchain() is not None
        expected = "c" if has_compiler else "python"
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["target"], expected)

    def test_self_heal_patch_fixes_every_error(self) -> None:
        if shutil.which("git") is None:
            self.skipTest("git is required to verify unified patch application")

        checked = run_cli("check", "examples/self_heal.ae")
        self.assertEqual(checked.returncode, 1)
        payload = json.loads(checked.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertIn("fix_patch", payload)
        self.assertGreaterEqual(len(payload["diagnostics"]), 4)

        directory = pathlib.Path(tempfile.mkdtemp(prefix="aether_heal_"))
        (directory / "examples").mkdir(parents=True)
        target = directory / "examples" / "self_heal.ae"
        shutil.copy(ROOT / "examples" / "self_heal.ae", target)
        patch_file = directory / "fix.patch"
        patch_file.write_text(payload["fix_patch"], encoding="utf-8", newline="")

        applied = subprocess.run(
            ["git", "apply", "fix.patch"],
            cwd=directory,
            capture_output=True,
            text=True,
        )
        self.assertEqual(applied.returncode, 0, applied.stderr)

        healed = run_cli("check", str(target))
        healed_payload = json.loads(healed.stdout)
        self.assertEqual(healed_payload["status"], "ok")
        self.assertEqual(healed_payload["summary"]["errors"], 0)

        text = target.read_text(encoding="utf-8")
        self.assertIn("use ai.core", text)
        self.assertIn("-> auto_hardware", text)
        self.assertIn("-> optimize(speed: balanced)", text)
        self.assertIn("-> stream_output", text)

    def test_type_mismatch_is_a_diagnostic(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task Bad:\n"
            "    a = tensor(64, 32) -> auto_hardware\n"
            "    b = tensor(64, 16) -> auto_hardware\n"
            "    net = neural_net(in: a, out: 8) -> optimize(speed: max)\n"
            "    run net(b) -> stream_output\n"
        )
        result = run_cli("check", str(path))
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(any(d["code"] == "AE0210" for d in payload["diagnostics"]))

    def test_tab_indentation_is_fixable(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "task Tabbed:\n"
            "\tx = tensor(8, 8) -> auto_hardware\n"
            "\trun x(x) -> stream_output\n"
        )
        result = run_cli("check", str(path))
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["diagnostics"][0]["code"], "AE0002")
        self.assertIn("    x = tensor", payload["fix_patch"])

    def test_unknown_intent_suggests_repair(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task Typo:\n"
            "    x = tensr(8, 8) -> auto_hardware\n"
        )
        result = run_cli("check", str(path))
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "error")
        diagnostic = next(d for d in payload["diagnostics"] if d["code"] == "AE0201")
        self.assertIn("tensor", diagnostic["hint"])
        self.assertIn("+    x = tensor(8, 8)", payload["fix_patch"])

    def test_task_selection_and_emit(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task First:\n"
            "    x = tensor(16, 8) -> auto_hardware\n"
            "\n"
            "task Second:\n"
            "    x = tensor(16, 8) -> auto_hardware\n"
            "    emit x\n"
        )
        checked = run_cli("check", str(path))
        self.assertEqual(checked.returncode, 0, checked.stdout)
        result = run_cli("run", str(path), "--task", "Second", "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(any(event["event"] == "emit_tensor" for event in payload["events"]))

    def test_missing_file_exit_code(self) -> None:
        result = run_cli("check", "does_not_exist.ae")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(payload["diagnostics"][0]["code"], "AE0000")

    def test_unknown_task_exit_code(self) -> None:
        result = run_cli("run", "hello_aether.ae", "--task", "Nope", "--json")
        payload = json.loads(result.stdout)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(payload["code"], "AE0503")

    # ------------------------------------------------------------------ v0.2
    def test_training_pipeline_runs_and_exports(self) -> None:
        target = "c" if native_toolchain() is not None else "python"
        result = run_cli(
            "run", "examples/train_ai_agent.ae", "--chunks", "4", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["target"], target)
        by_kind = {event["event"]: event for event in reversed(payload["events"])}
        self.assertIn("dataset", by_kind)
        self.assertIn("train_done", by_kind)
        self.assertGreaterEqual(by_kind["train_done"]["accuracy"], 0.85)
        self.assertIn("export", by_kind)
        self.assertEqual(by_kind["export"]["format"], "AEB1")
        self.assertTrue(pathlib.Path(by_kind["export"]["path"]).is_file())
        self.assertIn("task_done", by_kind)
        classifier_chunks = [
            event for event in payload["events"] if event["event"] == "chunk"
        ]
        self.assertEqual(len(classifier_chunks), 4)
        self.assertIn("accuracy", classifier_chunks[0])

    def test_training_works_on_python_backend(self) -> None:
        result = run_cli(
            "run",
            "examples/train_ai_agent.ae",
            "--chunks",
            "2",
            "--target",
            "python",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "python")
        train_done = next(
            event for event in reversed(payload["events"]) if event["event"] == "train_done"
        )
        self.assertGreaterEqual(train_done["accuracy"], 0.85)
        self.assertEqual(train_done["epochs"], 5)
        export = next(
            event for event in reversed(payload["events"]) if event["event"] == "export"
        )
        self.assertEqual(export["bytes"], 32 + 17802 * 4)

    def test_classifier_type_errors_are_diagnostics(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task Mismatch:\n"
            "    dataset = load_data(samples: 1000, features: 32) -> auto_hardware\n"
            "    model = classifier(in: 32, classes: 4) -> optimize(speed: max)\n"
            "    train model on dataset(epochs: 2) -> self_heal\n"
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        mismatch = next(d for d in payload["diagnostics"] if d["code"] == "AE0210")
        self.assertIn("Class mismatch", mismatch["message"])
        self.assertIn("classes: 10", payload["fix_patch"])

    def test_train_requires_self_heal_modifier(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task NoHeal:\n"
            "    dataset = load_data(samples: 1000, features: 32) -> auto_hardware\n"
            "    model = classifier(in: 32, classes: 10) -> optimize(speed: max)\n"
            "    train model on dataset(epochs: 2)\n"
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(
            any(d["code"] == "AE0300" for d in payload["diagnostics"]),
            payload["diagnostics"],
        )
        self.assertIn("-> self_heal", payload["fix_patch"])

    def test_export_before_train_is_rejected(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task EarlyExport:\n"
            "    dataset = load_data(samples: 1000, features: 32) -> auto_hardware\n"
            "    model = classifier(in: 32, classes: 10) -> optimize(speed: max)\n"
            "    export model -> \"model.ae.bin\"\n"
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(
            any("must be trained" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

    def test_bench_reports_comparison(self) -> None:
        from aetherc.cbackend import find_toolchain

        args = ["bench", "examples/train_ai_agent.ae", "--repeat", "1", "--json"]
        result = run_cli(*args)
        self.assertEqual(result.returncode, 0, result.stdout[-3000:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["workload"]["samples"], 10000)
        self.assertEqual(payload["workload"]["epochs"], 5)
        names = {row["name"]: row for row in payload["rows"]}
        self.assertIn("aether -> Python/NumPy", names)
        self.assertIn("python/numpy (hand-written)", names)
        python_row = names["aether -> Python/NumPy"]
        self.assertIsNotNone(python_row["ms"])
        self.assertGreater(python_row["ms"], 0)
        if find_toolchain() is not None:
            native_row = names["aether -> C/OpenMP (native)"]
            self.assertIsNone(native_row.get("note"))
            self.assertIsNotNone(native_row["ms"])
        aether_tokens = python_row["tokens"]
        numpy_tokens = names["python/numpy (hand-written)"]["tokens"]
        self.assertLess(aether_tokens, numpy_tokens * 0.25)

    def test_workload_extraction_matches_example(self) -> None:
        from aetherc.bench import extract_workload
        from aetherc.lexer import tokenize
        from aetherc.parser import parse

        source = (ROOT / "examples" / "train_ai_agent.ae").read_text(encoding="utf-8-sig")
        program = parse(tokenize(source, "train_ai_agent.ae"), "train_ai_agent.ae", source)
        workload = extract_workload(program, "TrainAgent")
        self.assertEqual(workload.samples, 10000)
        self.assertEqual(workload.features, 128)
        self.assertEqual(workload.classes, 10)
        self.assertEqual(workload.hidden, 128)
        self.assertEqual(workload.epochs, 5)


if __name__ == "__main__":
    unittest.main()
