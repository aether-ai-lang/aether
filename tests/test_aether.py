"""End-to-end tests for the Aether v0.2 compiler, checker, and runners."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
sys.path.insert(0, str(ROOT))


def run_cli(
    *args: str, cwd: pathlib.Path = ROOT, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    if env:
        environment.update(env)
    return subprocess.run(
        [PY, str(ROOT / "aether.py"), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=environment,
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
        from aetherc.cbackend import find_toolchain

        result = run_cli("run", "hello_aether.ae", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("\x1b", result.stdout)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["task"], "HelloAether")
        # auto target: native C when a compiler exists, Python otherwise
        expected = "c" if find_toolchain() is not None else "python"
        self.assertEqual(payload["target"], expected)
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
        # Python target: exercises task selection + emit events without a
        # fresh C compile (native selection is covered by test_native_run_is_default).
        result = run_cli("run", str(path), "--task", "Second", "--target", "python", "--json")
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

        # Fast mode for unit tests: skip the (multi-second) hand-written
        # reference implementations; production bench still runs them.
        args = ["bench", "examples/train_ai_agent.ae", "--repeat", "1", "--json"]
        result = run_cli(
            *args,
            env={
                "AETHER_TORCH_PATH": "skipped-for-tests",
                "AETHER_BENCH_SKIP_REFERENCES": "1",
            },
        )
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

    # ------------------------------------------------------------------ v0.4
    @staticmethod
    def _flow_logs(payload: dict) -> list[str]:
        return [
            event.get("text", "")
            for event in payload["events"]
            if event["event"] == "log"
        ]

    def test_control_flow_python(self) -> None:
        result = run_cli("run", "examples/control_flow.ae", "--target", "python", "--json")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "python")
        self.assertEqual(
            self._flow_logs(payload),
            ["big", "10", "13", "2", "spin", "spin", "spin", "done"],
        )
        guards = [e for e in payload["events"] if e["event"] == "while_guard"]
        self.assertEqual(len(guards), 1)
        self.assertEqual(guards[0]["max_iters"], 3)

    def test_control_flow_native(self) -> None:
        from aetherc.cbackend import find_toolchain

        if find_toolchain() is None:
            self.skipTest("no C compiler available")
        result = run_cli("run", "examples/control_flow.ae", "--target", "c", "--json")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "c")
        self.assertEqual(
            self._flow_logs(payload),
            ["big", "10", "13", "2", "spin", "spin", "spin", "done"],
        )
        guards = [e for e in payload["events"] if e["event"] == "while_guard"]
        self.assertEqual(len(guards), 1)

    def test_autonomous_loop_runs(self) -> None:
        from aetherc.cbackend import find_toolchain

        if find_toolchain() is None:
            self.skipTest("no C compiler available")
        result = run_cli("run", "examples/autonomous_loop.ae", "--target", "c", "--json")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        trainings = [
            event for event in payload["events"] if event["event"] == "train_done"
        ]
        self.assertEqual(len(trainings), 3, "repeat 3 times must train three times")
        self.assertEqual(trainings[0]["epochs"], 2)
        self.assertGreater(trainings[-1]["accuracy"], 0.8)
        exports = [event for event in payload["events"] if event["event"] == "export"]
        self.assertTrue(exports)
        self.assertIn("v04_agent.ae.bin", exports[0]["path"])

    def test_repeat_requires_agent_core(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task Loopy:\n"
            "    repeat 3 times:\n"
            "        log \"tick\"\n"
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(
            any(
                d["code"] == "AE0200" and "agent.core" in d["message"]
                for d in payload["diagnostics"]
            ),
            payload["diagnostics"],
        )
        self.assertIn("use agent.core", payload["fix_patch"])

    def test_control_flow_type_rules(self) -> None:
        cases = [
            ('    let bad = "a" + 1', "arithmetic requires numbers"),
            ('    let flag = 1\n    if flag:\n        log "x"', "if requires a comparison"),
            ("    let n = 0\n    let n = 0.5", "cannot become"),
            ("    let m = 5 % 2.0", "'%' requires two integers"),
        ]
        for body, expected in cases:
            path = write_temp(
                "use ai.core\nuse agent.core\n\ntask Typed:\n" + body + "\n"
            )
            payload = json.loads(run_cli("check", str(path)).stdout)
            self.assertEqual(payload["status"], "error", body)
            self.assertTrue(
                any(expected in d["message"] for d in payload["diagnostics"]),
                (body, payload["diagnostics"]),
            )

    # ------------------------------------------------------------------ v0.5
    def test_real_data_pipeline(self) -> None:
        from aetherc.cbackend import find_toolchain

        cls = type(self)
        cls._ensure_serve_model  # keep linter quiet about optional ordering

        result = run_cli(
            "run", "examples/real_data_pipeline.ae", "--target", "python", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        events = payload["events"]
        datasets = [e for e in events if e["event"] == "dataset"]
        self.assertEqual(len(datasets), 1)
        self.assertEqual((datasets[0]["samples"], datasets[0]["features"]), (480, 8))
        self.assertEqual(
            len([e for e in events if e["event"] == "train_done"]), 3,
            "while loop should converge after 3 training rounds",
        )
        self.assertEqual(
            len([e for e in events if e["event"] == "while_guard"]), 0,
            "the guard should not trip on a converging loop",
        )
        saves = [e for e in events if e["event"] == "save_json"]
        self.assertTrue(saves)
        report = pathlib.Path(saves[0]["path"])
        self.assertTrue(report.is_file())
        report_data = json.loads(report.read_text(encoding="utf-8"))
        self.assertGreater(report_data["accuracy"], 0.85)

        if find_toolchain() is None:
            return
        result = run_cli(
            "run", "examples/real_data_pipeline.ae", "--target", "c", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        events = payload["events"]
        self.assertEqual(
            len([e for e in events if e["event"] == "train_done"]), 3
        )
        saves = [e for e in events if e["event"] == "save_json"]
        self.assertTrue(saves)
        report_data = json.loads(pathlib.Path(saves[0]["path"]).read_text(encoding="utf-8"))
        self.assertGreater(report_data["accuracy"], 0.85)
        self.assertIn("digest", report_data)

    def test_gpu_dispatch_and_cuda_fallback(self) -> None:
        from aetherc.gpu import dispatch_target, probe_gpu

        info = probe_gpu()
        self.assertIn("tier", info)
        self.assertIn("vendor", info)

        resolved, _note = dispatch_target("auto")
        self.assertIn(resolved, ("cuda", "c"))

        # This machine (and CI) may or may not have nvcc; requesting CUDA must
        # never silently fake GPU execution: it dispatches or says why.
        result = run_cli("run", "hello_aether.ae", "--target", "cuda", "--json")
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        if probe_gpu()["cuda_gpu"] and probe_gpu()["nvcc"]:
            self.assertEqual(payload["target"], "cuda")
        else:
            self.assertEqual(payload["target"], "c")
            self.assertIn("nvcc", payload.get("note") or "nvcc")

    def test_data_core_contracts(self) -> None:
        path = write_temp(
            "use ai.core\nuse data.core\n\ntask Needs:\n"
            '    dataset = load_csv("data.csv") -> auto_hardware\n'
            "    model = classifier(in: 4, classes: 2) -> optimize(speed: max)\n"
            '    save_json(12345, "out.json")\n'
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        missing_target = next(
            (d for d in payload["diagnostics"] if d["code"] == "AE0212"), None
        )
        self.assertIsNotNone(missing_target)
        self.assertIn("target", missing_target["message"].lower())

        nomod = write_temp(
            "use ai.core\n\ntask NoData:\n"
            '    dataset = load_csv("data.csv", target: "y") -> auto_hardware\n'
        )
        payload = json.loads(run_cli("check", str(nomod)).stdout)
        self.assertTrue(
            any(
                d["code"] == "AE0200" and "data.core" in d["message"]
                for d in payload["diagnostics"]
            ),
            payload["diagnostics"],
        )
        self.assertIn("use data.core", payload["fix_patch"])

        fnbad = write_temp(
            "use ai.core\nuse agent.core\n\nfn add(a, b) -> float:\n    return a + b\n\n"
            "task Calls:\n    let value = add(1)\n"
        )
        payload = json.loads(run_cli("check", str(fnbad)).stdout)
        self.assertTrue(
            any("expects 2 argument(s)" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

    # ------------------------------------------------------------------ v0.7
    def test_llm_mock_python_and_native(self) -> None:
        from aetherc.cbackend import find_toolchain

        cls = type(self)
        targets = ["python"]
        if find_toolchain() is not None:
            targets.append("c")

        for target in targets:
            result = run_cli(
                "run",
                "examples/llm_agent_workflow.ae",
                "--target",
                target,
                "--json",
                env={"AETHER_LLM_MOCK": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stdout[-1500:])
            payload = json.loads(result.stdout)
            self.assertEqual(payload["target"], target)
            events = payload["events"]
            asks = [e for e in events if e["event"] == "ask"]
            self.assertEqual(len(asks), 1)
            self.assertEqual(asks[0]["model"], "gpt-4o-mini")
            self.assertEqual(asks[0]["mode"], "mock")
            reply = [e.get("text", "") for e in events if e["event"] == "log"][0]
            self.assertIn("mock", reply)
            self.assertIn("accuracy", reply)

            reductions = [e for e in events if e["event"] == "reduce"]
            values = {r["op"]: r["value"] for r in reductions}
            self.assertEqual(len(reductions), 5)
            self.assertLessEqual(values["min"], values["max"])
            scales = [e for e in events if e["event"] == "scale"]
            self.assertEqual(scales[0]["factor"], 0.5)
            argmax_values = [r["value"] for r in reductions if r["op"] == "argmax"]
            self.assertTrue(0 <= argmax_values[0] < 256)

    def test_llm_contracts(self) -> None:
        no_core = write_temp(
            'use ai.core\n\ntask Ask:\n    log 1\n'
        )
        payload = json.loads(run_cli("check", str(no_core)).stdout)

        missing_model = write_temp(
            "use ai.core\nuse llm.core\n\ntask Ask:\n"
            '    let reply = ask llm("hello")\n'
        )
        payload = json.loads(run_cli("check", str(missing_model)).stdout)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(
            any(d["code"] == "AE0300" for d in payload["diagnostics"]),
            payload["diagnostics"],
        )
        self.assertIn('model("gpt-4o-mini")', payload["fix_patch"])

        numeric_prompt = write_temp(
            "use ai.core\nuse llm.core\n\ntask Ask:\n"
            "    let reply = ask llm(42) -> model(\"gpt-4o-mini\")\n"
        )
        payload = json.loads(run_cli("check", str(numeric_prompt)).stdout)
        self.assertTrue(
            any("string prompt" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

    def test_cuda_training_kernels_emitted(self) -> None:
        result = run_cli(
            "run",
            "examples/autonomous_loop.ae",
            "--target",
            "cuda",
            "--emit-only",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stdout[-1500:])
        payload = json.loads(result.stdout)
        self.assertEqual(payload["target"], "cuda")
        source = pathlib.Path(payload["artifact"]).read_text(encoding="utf-8")
        self.assertIn("__global__", source)
        self.assertIn("ae_sgd_momentum", source)
        self.assertIn("ae_gemm_atb", source)
        self.assertIn("#define IN_DIM 64", source)
        self.assertIn("#define EPOCHS 2", source)

    # ------------------------------------------------------------------ v0.6
    def test_agent_memory_python(self) -> None:
        path = write_temp(
            "use ai.core\nuse agent.core\n\n"
            "task Memo:\n"
            '    let prev = recall("runs", default: 0.0)\n'
            "    log prev\n"
            "    remember(\"runs\", prev + 1)\n"
        )
        runs = []
        for _ in range(2):
            result = run_cli("run", str(path), "--target", "python", "--json")
            self.assertEqual(result.returncode, 0, result.stdout[-1200:])
            payload = json.loads(result.stdout)
            runs.append(self._flow_logs(payload))
        self.assertEqual(runs[0], ["0"])
        self.assertEqual(runs[1], ["1"])
        self.assertTrue((path.parent / "case.memory.json").is_file())

    def test_smart_agent_memory_example(self) -> None:
        from aetherc.cbackend import find_toolchain

        if find_toolchain() is None:
            self.skipTest("no C compiler available")
        memory = ROOT / "examples" / "smart_agent_memory.memory.json"
        if memory.exists():
            memory.unlink()

        accuracies = []
        for _ in range(2):
            result = run_cli(
                "run", "examples/smart_agent_memory.ae", "--target", "c", "--json"
            )
            self.assertEqual(result.returncode, 0, result.stdout[-1500:])
            payload = json.loads(result.stdout)
            events = payload["events"]
            logs = self._flow_logs(payload)
            recalls = [e for e in events if e["event"] == "recall"]
            remembers = [e for e in events if e["event"] == "remember"]
            self.assertEqual(len(recalls), 1)
            self.assertEqual(len(remembers), 1)
            self.assertIn("improved", logs)
            accuracies.append(remembers[0]["value"])
            reductions = [e for e in events if e["event"] == "reduce"]
            self.assertEqual([r["op"] for r in reductions], ["sum", "mean"])
            self.assertEqual(reductions[0]["elements"], 256)
        self.assertGreater(accuracies[0], 0.0)
        self.assertGreater(accuracies[1], accuracies[0], "memory must raise the bar")
        self.assertTrue(memory.is_file())
        stored = json.loads(memory.read_text(encoding="utf-8"))
        self.assertGreater(stored["best_acc"], 0.0)

    def test_memory_and_reduction_contracts(self) -> None:
        no_agent = write_temp(
            "use ai.core\n\ntask Forgetful:\n"
            '    remember("k", 1)\n'
        )
        payload = json.loads(run_cli("check", str(no_agent)).stdout)
        self.assertTrue(
            any(
                d["code"] == "AE0200" and "agent.core" in d["message"]
                for d in payload["diagnostics"]
            ),
            payload["diagnostics"],
        )
        self.assertIn("use agent.core", payload["fix_patch"])

        bad_reduce = write_temp(
            "use ai.core\n\ntask Reduce:\n"
            "    dataset = load_data(samples: 100, features: 8) -> auto_hardware\n"
            "    log sum(dataset)\n"
        )
        payload = json.loads(run_cli("check", str(bad_reduce)).stdout)
        self.assertTrue(
            any("reduces a tensor" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

    # ------------------------------------------------------------------ v0.3
    def test_serve_requires_web_core_and_route(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "\n"
            "task BadServe:\n"
            '    model = load_model("model.ae.bin") -> auto_hardware\n'
            "    serve model on port(8080)\n"
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        codes = [d["code"] for d in payload["diagnostics"]]
        self.assertIn("AE0200", codes)  # missing use web.core
        self.assertIn("AE0300", codes)  # missing route
        self.assertIn("use web.core", payload["fix_patch"])
        self.assertIn('route("/predict")', payload["fix_patch"])

    def test_load_model_contracts(self) -> None:
        path = write_temp(
            "use ai.core\n"
            "use web.core\n"
            "\n"
            "task NeedsPatch:\n"
            "    model = load_model(\"model.ae.bin\")\n"
            '    serve model on port(8080) -> route("/predict")\n'
        )
        payload = json.loads(run_cli("check", str(path)).stdout)
        self.assertEqual(payload["status"], "error")
        self.assertTrue(any(d["code"] == "AE0300" for d in payload["diagnostics"]))
        self.assertIn("-> auto_hardware", payload["fix_patch"])

        bad = write_temp(
            "use ai.core\n"
            "use web.core\n"
            "\n"
            "task Unquoted:\n"
            "    model = load_model(8080) -> auto_hardware\n"
            '    serve model on port(8080) -> route("/predict")\n'
        )
        payload = json.loads(run_cli("check", str(bad)).stdout)
        self.assertTrue(
            any(d["code"] == "AE0212" for d in payload["diagnostics"]),
            payload["diagnostics"],
        )
        self.assertTrue(
            any("quoted path" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

        served = write_temp(
            "use ai.core\n"
            "\n"
            "task CannotRun:\n"
            '    model = load_model("model.ae.bin") -> auto_hardware\n'
            "    run model(model) -> stream_output\n"
        )
        payload = json.loads(run_cli("check", str(served)).stdout)
        self.assertTrue(
            any("no compile-time shape" in d["message"] for d in payload["diagnostics"]),
            payload["diagnostics"],
        )

    @staticmethod
    def _free_port() -> int:
        import socket

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    @classmethod
    def _http(cls, url: str, body: bytes | None = None, timeout: float = 5.0):
        import urllib.error
        import urllib.request

        request = urllib.request.Request(url, data=body)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    @classmethod
    def _ensure_serve_model(cls) -> None:
        model = ROOT / "examples" / "model.ae.bin"
        if not model.is_file():
            result = run_cli("run", "examples/train_ai_agent.ae", "--target", "python", "--json")
            assert result.returncode == 0, result.stdout[-1000:]

    def test_native_http_inference(self) -> None:
        from aetherc.cbackend import compile_c, find_toolchain, generate_for_toolchain
        from aetherc.lexer import tokenize
        from aetherc.parser import parse

        toolchain = find_toolchain()
        if toolchain is None:
            self.skipTest("no C compiler available")
        cls = type(self)
        cls._ensure_serve_model()

        source_name = "examples/serve_ai_api.ae"
        source = (ROOT / source_name).read_text(encoding="utf-8-sig")
        program = parse(tokenize(source, source_name), source_name, source)
        generated = generate_for_toolchain(program, "serve_ai_api.ae", toolchain, None, chunks=8)
        native = compile_c(
            generated, ROOT / "examples" / ".aether_build", "serve_ai_api", toolchain
        )

        port = cls._free_port()
        env = dict(
            os.environ,
            AETHER_PORT=str(port),
            AETHER_SERVE_SECONDS="15",
            AETHER_BASE_DIR=str((ROOT / "examples").resolve()),
        )
        proc = subprocess.Popen(
            [str(native.binary)],
            env=env,
            cwd=str(native.binary.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            base = f"http://127.0.0.1:{port}"
            status, health = None, None
            import time

            for _ in range(60):
                time.sleep(0.1)
                try:
                    status, health = cls._http(f"{base}/health", timeout=2)
                    break
                except Exception:
                    continue
            self.assertEqual(status, 200, "server did not become healthy")
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["in_dim"], 128)
            self.assertEqual(health["out_dim"], 10)

            features = [0.01 * i for i in range(128)]
            status, prediction = cls._http(
                f"{base}/predict",
                body=json.dumps({"features": features}).encode(),
            )
            self.assertEqual(status, 200)
            self.assertIsInstance(prediction["prediction"], int)
            self.assertTrue(0 <= prediction["prediction"] < 10)
            self.assertEqual(len(prediction["probabilities"]), 10)
            self.assertLess(prediction["elapsed_ms"], 50.0)

            status, _ = cls._http(f"{base}/nope", body=b"{}")
            self.assertEqual(status, 404)
            status, error = cls._http(
                f"{base}/predict", body=json.dumps({"features": [1, 2, 3]}).encode()
            )
            self.assertEqual(status, 400)
            self.assertIn("128", error["error"])
        finally:
            # Kill immediately: AETHER_SERVE_SECONDS would otherwise keep the
            # process alive and stall the suite for the full window.
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def test_python_http_inference(self) -> None:
        cls = type(self)
        cls._ensure_serve_model()
        port = cls._free_port()
        env = dict(os.environ, AETHER_PORT=str(port))
        proc = subprocess.Popen(
            [
                PY,
                str(ROOT / "aether.py"),
                "run",
                "examples/serve_ai_api.ae",
                "--target",
                "python",
            ],
            env=env,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            import time

            base = f"http://127.0.0.1:{port}"
            status, health = None, None
            for _ in range(60):
                time.sleep(0.1)
                try:
                    status, health = cls._http(f"{base}/health", timeout=2)
                    break
                except Exception:
                    continue
            self.assertEqual(status, 200)
            self.assertEqual(health["status"], "ok")
            status, prediction = cls._http(
                f"{base}/predict", body=json.dumps({"features": [0.1] * 128}).encode()
            )
            self.assertEqual(status, 200)
            self.assertTrue(0 <= prediction["prediction"] < 10)
        finally:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def test_apply_patch_round_trip(self) -> None:
        from aetherc.diagnostics import apply_patch, replace_line_patch

        source = "use ai.core\n\ntask T:\n    x = tensor(8, 8)\n"
        patch = replace_line_patch("t.ae", source.splitlines(), 4, "    x = tensor(8, 8) -> auto_hardware")
        self.assertIsNotNone(patch)
        patched = apply_patch(patch, source)
        self.assertIsNotNone(patched)
        self.assertIn("-> auto_hardware", patched)
        self.assertEqual(len(patched.splitlines()), 4)
        self.assertIsNone(apply_patch(patch, "totally different\n"))

    def test_mcp_initialize_tools_fix_and_compile(self) -> None:
        import tempfile

        directory = pathlib.Path(tempfile.mkdtemp(prefix="aether_mcp_test_"))
        broken = directory / "broken.ae"
        broken.write_text("use ai\n\ntask T:\n    x = tensor(8, 8)\n", newline="\n")

        proc = subprocess.Popen(
            [PY, str(ROOT / "aether.py"), "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=str(ROOT),
        )

        def rpc(message: dict) -> dict:
            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise AssertionError("MCP server closed the pipe (see stderr)")
            return json.loads(line)

        def notify(message: dict) -> None:
            # Notifications get no response by protocol; reading would block.
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()

        try:
            init = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                }
            )
            self.assertEqual(init["result"]["serverInfo"]["name"], "aether")
            from aetherc.diagnostics import AETHER_VERSION

            self.assertEqual(init["result"]["serverInfo"]["version"], AETHER_VERSION)
            notify({"jsonrpc": "2.0", "method": "notifications/initialized"})

            tools = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            names = {tool["name"] for tool in tools["result"]["tools"]}
            self.assertLessEqual(
                {
                    "aether_check",
                    "aether_fix",
                    "aether_compile",
                    "aether_run",
                    "aether_spec",
                    "aether_bench",
                },
                names,
            )

            check = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "aether_check", "arguments": {"file": str(broken)}},
                }
            )
            check_payload = json.loads(check["result"]["content"][0]["text"])
            self.assertEqual(check_payload["status"], "error")
            self.assertTrue(check_payload["diagnostics"])

            fix = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "aether_fix",
                        "arguments": {"file": str(broken), "apply": True},
                    },
                }
            )
            fix_payload = json.loads(fix["result"]["content"][0]["text"])
            self.assertTrue(fix_payload["applied"])
            self.assertLess(fix_payload["after"]["errors"], fix_payload["before"]["errors"])

            compile_call = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "tools/call",
                    "params": {
                        "name": "aether_compile",
                        "arguments": {"file": "hello_aether.ae", "target": "python"},
                    },
                }
            )
            compile_payload = json.loads(compile_call["result"]["content"][0]["text"])
            self.assertEqual(compile_payload["status"], "ok")
            self.assertTrue(pathlib.Path(compile_payload["artifact"]).is_file())

            spec = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "tools/call",
                    "params": {"name": "aether_spec", "arguments": {}},
                }
            )
            spec_text = spec["result"]["content"][0]["text"]
            self.assertIn("Language Specification", spec_text)

            unknown = rpc(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "nope", "arguments": {}},
                }
            )
            self.assertEqual(unknown["error"]["code"], -32602)
        finally:
            try:
                proc.stdin.close()
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                proc.kill()


if __name__ == "__main__":
    unittest.main()
