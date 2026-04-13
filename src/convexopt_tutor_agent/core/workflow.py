from __future__ import annotations

from typing import Callable, Iterable

from convexopt_tutor_agent.core.prompting import build_analysis_prompts, build_code_prompts
from convexopt_tutor_agent.core.schema import (
    AppSettings,
    BuiltinExample,
    CodeBundle,
    ConstraintSpec,
    DataSymbol,
    ExecutionSummary,
    ProblemAnalysis,
    VariableSpec,
    WorkflowState,
)
from convexopt_tutor_agent.execution.local_runner import LocalExecutionRunner
from convexopt_tutor_agent.llm.kimi_adapter import KimiClient

ProgressCallback = Callable[[str], None]


class UserFacingError(RuntimeError):
    pass


class TutorWorkflowService:
    def __init__(
        self,
        kimi_client: KimiClient,
        execution_runner: LocalExecutionRunner,
        examples: Iterable[BuiltinExample],
    ) -> None:
        self.kimi_client = kimi_client
        self.execution_runner = execution_runner
        self.examples = {example.key: example for example in examples}

    def analyze_problem(
        self,
        input_text: str,
        settings: AppSettings,
        example_key: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> WorkflowState:
        input_text = input_text.strip()
        if not input_text:
            raise UserFacingError("Please enter an optimization problem first.")

        logs: list[str] = []
        emit = _progress_emitter(progress, logs)

        if example_key and example_key in self.examples:
            emit("Using a built-in example. No Kimi request is needed.")
            example = self.examples[example_key]
            code = self.execution_runner.prepare_code_bundle(example.model_code, notes="Built-in example code.")
            return WorkflowState(
                input_text=input_text,
                example_key=example_key,
                analysis=example.analysis,
                code=code,
                logs=logs,
            )

        if not settings.api_key:
            raise UserFacingError("Free-text analysis requires a Moonshot API Key in Settings.")

        emit("Sending a structured parsing request to Kimi.")
        analysis_system, analysis_user = build_analysis_prompts(input_text, settings)
        raw_analysis = self.kimi_client.complete_json(
            system_prompt=analysis_system,
            user_prompt=analysis_user,
            settings=settings,
            progress=emit,
        )
        analysis = self._analysis_from_json(raw_analysis)
        emit(f"Detected problem family: {analysis.problem_family}")

        code_bundle: CodeBundle | None = None
        if analysis.is_convex is True:
            emit("Problem identified as convex. Generating CVXPY code.")
            code_system, code_user = build_code_prompts(input_text, analysis, settings)
            raw_code = self.kimi_client.complete_json(
                system_prompt=code_system,
                user_prompt=code_user,
                settings=settings,
                progress=emit,
            )
            code_bundle = self._code_from_json(raw_code)
            if not code_bundle.model_code.strip():
                raise UserFacingError("Kimi did not return executable CVXPY code.")
            code_bundle = self.execution_runner.prepare_code_bundle(
                code_bundle.model_code,
                notes=code_bundle.notes,
                assumptions=code_bundle.assumptions,
                uses_synthesized_data=code_bundle.uses_synthesized_data,
            )
            emit("CVXPY code generated.")
        elif analysis.is_convex is False:
            emit("Problem identified as non-convex. Auto-solving is disabled.")
        else:
            emit("Convexity is still uncertain. Stopping at the analysis stage.")

        return WorkflowState(
            input_text=input_text,
            example_key=example_key,
            analysis=analysis,
            code=code_bundle,
            logs=logs,
        )

    def execute_solution(
        self,
        workflow: WorkflowState,
        settings: AppSettings,
        progress: ProgressCallback | None = None,
    ) -> ExecutionSummary:
        if workflow.analysis is None or workflow.code is None:
            raise UserFacingError("Please finish the analysis and generate CVXPY code first.")
        if workflow.analysis.is_convex is not True:
            raise UserFacingError("This problem is not a confirmed convex problem and cannot be executed automatically.")

        logs: list[str] = []
        emit = _progress_emitter(progress, logs)
        emit("Starting local execution in a controlled temporary workspace.")

        summary = self.execution_runner.run(
            workflow.code,
            timeout_seconds=settings.execution_timeout_seconds,
        )
        emit(f"Execution finished. Solver status: {summary.status}")
        if summary.solver_name:
            emit(f"Solver used: {summary.solver_name}")
        return summary

    def _analysis_from_json(self, payload: dict) -> ProblemAnalysis:
        variables = [
            VariableSpec(
                name=str(item.get("name", "")).strip() or "x",
                shape=str(item.get("shape", "scalar")).strip() or "scalar",
                domain=str(item.get("domain", "real")).strip() or "real",
                attributes=[str(value) for value in item.get("attributes", []) if str(value).strip()],
                description=str(item.get("description", "")).strip(),
            )
            for item in payload.get("variables", [])
            if isinstance(item, dict)
        ]
        constraints = [
            ConstraintSpec(
                label=str(item.get("label", f"c{index + 1}")).strip() or f"c{index + 1}",
                expression=str(item.get("expression", "")).strip(),
                kind=str(item.get("kind", "other")).strip() or "other",
                explanation=str(item.get("explanation", "")).strip(),
            )
            for index, item in enumerate(payload.get("constraints", []))
            if isinstance(item, dict)
        ]
        data_symbols = [
            DataSymbol(
                name=str(item.get("name", "")).strip(),
                role=str(item.get("role", "")).strip(),
                shape=str(item.get("shape", "")).strip(),
                provided=bool(item.get("provided", False)),
                value_repr=str(item.get("value_repr", "")).strip(),
            )
            for item in payload.get("data_symbols", [])
            if isinstance(item, dict)
        ]

        return ProblemAnalysis(
            source="llm",
            title=str(payload.get("title", "")).strip(),
            problem_family=str(payload.get("problem_family", "Unknown")).strip() or "Unknown",
            objective_sense=str(payload.get("objective", {}).get("sense", "minimize")).strip() or "minimize",
            objective_expression=str(payload.get("objective", {}).get("expression", "")).strip(),
            variables=variables,
            constraints=constraints,
            is_convex=_coerce_optional_bool(payload.get("is_convex")),
            convexity_summary=str(payload.get("convexity_summary", "")).strip(),
            convexity_reason=str(payload.get("convexity_reason", "")).strip(),
            modeling_notes=str(payload.get("modeling_notes", "")).strip(),
            result_interpretation=str(payload.get("result_interpretation", "")).strip(),
            data_status=str(payload.get("data_status", "missing")).strip() or "missing",
            data_symbols=data_symbols,
            assumptions=[str(item).strip() for item in payload.get("assumptions", []) if str(item).strip()],
            synthesis_allowed=bool(payload.get("synthesis_allowed", False)),
            synthesis_notes=str(payload.get("synthesis_notes", "")).strip(),
        )

    def _code_from_json(self, payload: dict) -> CodeBundle:
        should_generate = bool(payload.get("should_generate_code", False))
        if not should_generate:
            raise UserFacingError(str(payload.get("notes", "Unable to generate CVXPY code.")))
        return CodeBundle(
            model_code=str(payload.get("model_code", "")).strip(),
            notes=str(payload.get("notes", "")).strip(),
            assumptions=[str(item).strip() for item in payload.get("assumptions", []) if str(item).strip()],
            uses_synthesized_data=bool(payload.get("uses_synthesized_data", False)),
        )


def _coerce_optional_bool(value: object) -> bool | None:
    if value is True or value is False:
        return bool(value)
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _progress_emitter(progress: ProgressCallback | None, logs: list[str]) -> ProgressCallback:
    def emit(message: str) -> None:
        logs.append(message)
        if progress is not None:
            progress(message)

    return emit
