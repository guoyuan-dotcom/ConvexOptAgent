from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

from convexopt_tutor_agent.core.schema import CodeBundle, ExecutionSummary

RESULT_START = "__CONVEXOPT_TUTOR_RESULT_START__"
RESULT_END = "__CONVEXOPT_TUTOR_RESULT_END__"
WORKER_MODE_ARG = "--worker-run"


@dataclass(slots=True)
class ExecutionPolicy:
    workspace_root: Path | None = None


class LocalExecutionRunner:
    def __init__(self, policy: ExecutionPolicy | None = None) -> None:
        self.policy = policy or ExecutionPolicy()

    def prepare_code_bundle(
        self,
        model_code: str,
        *,
        notes: str = "",
        assumptions: list[str] | None = None,
        uses_synthesized_data: bool = False,
    ) -> CodeBundle:
        executable_code = self._build_executable_script(model_code)
        return CodeBundle(
            model_code=model_code.strip(),
            executable_code=executable_code,
            notes=notes,
            assumptions=assumptions or [],
            uses_synthesized_data=uses_synthesized_data,
        )

    def run(self, code_bundle: CodeBundle, timeout_seconds: int = 20) -> ExecutionSummary:
        self._validate_model_code(code_bundle.model_code)

        workspace_root = self.policy.workspace_root or Path(tempfile.gettempdir())
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = Path(
            tempfile.mkdtemp(prefix="convexopt_tutor_agent_", dir=str(workspace_root))
        )
        script_path = workspace / "generated_problem.py"
        result_path = workspace / "worker_result.json"
        script_path.write_text(code_bundle.executable_code, encoding="utf-8")

        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"

        run_kwargs = {
            "args": self._build_worker_command(script_path, result_path),
            "cwd": str(workspace),
            "capture_output": True,
            "text": True,
            "timeout": timeout_seconds,
            "shell": False,
            "env": env,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        started_at = time.perf_counter()
        try:
            completed = subprocess.run(**run_kwargs)
        except subprocess.TimeoutExpired as exc:
            return ExecutionSummary(
                status="Timeout",
                stdout=(exc.stdout or "").strip(),
                stderr=(exc.stderr or "").strip(),
                workspace=str(workspace),
                duration_seconds=time.perf_counter() - started_at,
                result_note=f"Execution exceeded {timeout_seconds} seconds and was terminated.",
            )

        duration = time.perf_counter() - started_at
        worker_result = self._load_worker_result(result_path)
        stdout = str(worker_result.get("stdout", completed.stdout or ""))
        stderr = str(worker_result.get("stderr", completed.stderr or ""))
        error_message = str(worker_result.get("error", "")).strip()
        payload = worker_result.get("payload")
        clean_stdout = self._strip_result_markers(stdout)

        if not isinstance(payload, dict):
            status = "Execution Error" if completed.returncode or error_message else "Unknown"
            return ExecutionSummary(
                status=status,
                stdout=clean_stdout.strip(),
                stderr=stderr.strip(),
                workspace=str(workspace),
                duration_seconds=duration,
                result_note=error_message or "No structured solver output could be parsed from execution.",
            )

        return ExecutionSummary(
            status=str(payload.get("status", "Unknown")),
            solver_name=str(payload.get("solver_name", "")),
            optimal_value=str(payload.get("optimal_value", "N/A")),
            variable_values={
                str(name): self._value_to_text(value)
                for name, value in payload.get("variable_values", {}).items()
            },
            dual_values={
                str(name): self._value_to_text(value)
                for name, value in payload.get("dual_values", {}).items()
            },
            stdout=clean_stdout.strip(),
            stderr=stderr.strip(),
            workspace=str(workspace),
            duration_seconds=duration,
            result_note=str(payload.get("result_note", "")),
        )

    def _build_worker_command(self, script_path: Path, result_path: Path) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable, WORKER_MODE_ARG, str(script_path), str(result_path)]

        project_root = Path(__file__).resolve().parents[3]
        run_app_path = project_root / "run_app.py"
        return [
            sys.executable,
            "-I",
            "-B",
            str(run_app_path),
            WORKER_MODE_ARG,
            str(script_path),
            str(result_path),
        ]

    def _load_worker_result(self, result_path: Path) -> dict:
        if not result_path.exists():
            return {}
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _extract_payload(self, stdout: str) -> tuple[dict | None, str]:
        return _extract_payload_from_text(stdout)

    def _strip_result_markers(self, stdout: str) -> str:
        _, cleaned = self._extract_payload(stdout)
        return cleaned

    def _validate_model_code(self, code: str) -> None:
        tree = ast.parse(code, mode="exec")
        validator = _SafeModelCodeValidator()
        validator.visit(tree)
        validator.validate_required_symbols()

    def _build_executable_script(self, model_code: str) -> str:
        wrapper = textwrap.dedent(
            f"""
            import json
            import numpy as np
            import cvxpy as cp
            import warnings

            def _cota_normalize(value):
                if value is None:
                    return None
                if isinstance(value, dict):
                    return {{str(key): _cota_normalize(item) for key, item in value.items()}}
                if isinstance(value, (list, tuple)):
                    return [_cota_normalize(item) for item in value]
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                if hasattr(value, "tolist"):
                    try:
                        return value.tolist()
                    except Exception:
                        return str(value)
                if isinstance(value, (float, int, bool, str)):
                    return value
                return str(value)

            def _cota_to_named_mapping(collection, prefix):
                if collection is None:
                    return {{}}
                if isinstance(collection, dict):
                    return {{str(key): value for key, value in collection.items()}}
                if isinstance(collection, (list, tuple)):
                    mapping = {{}}
                    for index, item in enumerate(collection, start=1):
                        label = getattr(item, "name", None)
                        if callable(label):
                            try:
                                label = label()
                            except Exception:
                                label = None
                        mapping[str(label or f"{{prefix}}{{index}}")] = item
                    return mapping
                label = getattr(collection, "name", None)
                if callable(label):
                    try:
                        label = label()
                    except Exception:
                        label = None
                return {{str(label or prefix): collection}}

            def _cota_require_problem_instance(problem):
                if isinstance(problem, cp.Problem):
                    return problem
                raise TypeError(
                    "generated variable 'problem' is not a cvxpy.Problem instance; "
                    f"got {{type(problem).__name__}}"
                )

            def _cota_solve(problem):
                problem = _cota_require_problem_instance(problem)
                preferred = ["OSQP", "CLARABEL", "ECOS", "SCS"]
                installed = set(cp.installed_solvers())
                errors = []
                for name in preferred:
                    if name not in installed:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", FutureWarning)
                            problem.solve(solver=getattr(cp, name), verbose=False)
                        return name, errors
                    except Exception as exc:
                        errors.append(f"{{name}}: {{exc}}")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", FutureWarning)
                        problem.solve(verbose=False)
                except Exception as exc:
                    errors.append(f"default: {{exc}}")
                    raise RuntimeError("; ".join(errors))
                solver_name = ""
                if problem.solver_stats is not None and problem.solver_stats.solver_name:
                    solver_name = str(problem.solver_stats.solver_name)
                return solver_name or "default", errors

            try:
                _cota_solver_name, _cota_errors = _cota_solve(problem)
                _cota_tracked_variables = _cota_to_named_mapping(tracked_variables, "var_")
                _cota_tracked_constraints = _cota_to_named_mapping(tracked_constraints, "constraint_")
                _cota_variables = {{
                    str(name): _cota_normalize(getattr(var, "value", None))
                    for name, var in _cota_tracked_variables.items()
                }}
                _cota_duals = {{
                    str(name): _cota_normalize(getattr(constraint, "dual_value", None))
                    for name, constraint in _cota_tracked_constraints.items()
                }}
                _cota_result_note = f"{{data_summary}}\\n{{result_interpretation}}"
                if _cota_errors:
                    _cota_result_note += "\\nSolver fallback log: " + "; ".join(_cota_errors)
                _cota_payload = {{
                    "status": str(problem.status),
                    "solver_name": _cota_solver_name,
                    "optimal_value": _cota_normalize(problem.value),
                    "variable_values": _cota_variables,
                    "dual_values": _cota_duals,
                    "result_note": _cota_result_note,
                }}
            except Exception as exc:
                _cota_payload = {{
                    "status": "Execution Error",
                    "solver_name": "",
                    "optimal_value": None,
                    "variable_values": {{}},
                    "dual_values": {{}},
                    "result_note": f"Runtime error during execution: {{exc}}",
                }}

            print("{RESULT_START}")
            print(json.dumps(_cota_payload, ensure_ascii=False))
            print("{RESULT_END}")
            """
        ).strip()
        return f"{model_code.strip()}\n\n{wrapper}\n"

    def _value_to_text(self, value: object) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)


def run_worker_mode(script_path_arg: str, result_path_arg: str) -> int:
    script_path = Path(script_path_arg)
    result_path = Path(result_path_arg)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    payload: dict | None = None
    error_message = ""

    try:
        code = script_path.read_text(encoding="utf-8")
        globals_dict: dict[str, object] = {"__name__": "__main__", "__file__": str(script_path)}
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compile(code, str(script_path), "exec"), globals_dict, globals_dict)
        payload, _ = _extract_payload_from_text(stdout_buffer.getvalue())
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"

    result = {
        "payload": payload,
        "stdout": stdout_buffer.getvalue(),
        "stderr": stderr_buffer.getvalue(),
        "error": error_message,
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if not error_message else 1


def _extract_payload_from_text(stdout: str) -> tuple[dict | None, str]:
    pattern = re.compile(
        rf"{re.escape(RESULT_START)}\s*(\{{.*?\}})\s*{re.escape(RESULT_END)}",
        re.DOTALL,
    )
    match = pattern.search(stdout)
    if not match:
        return None, stdout
    payload = json.loads(match.group(1))
    cleaned = pattern.sub("", stdout)
    return payload, cleaned


class _SafeModelCodeValidator(ast.NodeVisitor):
    _allowed_imports = {"cvxpy", "numpy", "math"}
    _blocked_names = {
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "os",
        "sys",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "requests",
        "httpx",
        "importlib",
        "ctypes",
        "pickle",
    }
    _disallowed_nodes = (
        ast.ImportFrom,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
        ast.While,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )
    _required_names = {
        "problem",
        "tracked_variables",
        "tracked_constraints",
        "data_summary",
        "result_interpretation",
    }

    def __init__(self) -> None:
        self.assigned_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, self._disallowed_nodes):
            raise ValueError(f"Generated code contains a disallowed syntax node: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in self._allowed_imports:
                raise ValueError(f"Importing this module is not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self._blocked_names:
            raise ValueError(f"Generated code contains a blocked name: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        root_name = self._root_name(node)
        if root_name in self._blocked_names:
            raise ValueError(f"Generated code accesses a blocked object: {root_name}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        root_name = self._root_name(node.func)
        if root_name in self._blocked_names:
            raise ValueError(f"Generated code calls a blocked object: {root_name}")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record_target(node.target)
        self.generic_visit(node)

    def validate_required_symbols(self) -> None:
        missing = sorted(self._required_names - self.assigned_names)
        if missing:
            raise ValueError(f"Generated code is missing required symbols: {', '.join(missing)}")

    def _record_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.assigned_names.add(target.id)

    def _root_name(self, node: ast.AST) -> str | None:
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None
