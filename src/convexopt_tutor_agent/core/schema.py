from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class AppSettings:
    api_key: str = ""
    base_url: str = "https://api.moonshot.cn/v1"
    model: str = "moonshot-v1-8k"
    temperature: float = 0.2
    reasoning_mode: str = "default"
    request_timeout_seconds: int = 60
    execution_timeout_seconds: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class VariableSpec:
    name: str
    shape: str = "scalar"
    domain: str = "real"
    attributes: list[str] = field(default_factory=list)
    description: str = ""


@dataclass(slots=True)
class ConstraintSpec:
    label: str
    expression: str
    kind: str = "other"
    explanation: str = ""


@dataclass(slots=True)
class DataSymbol:
    name: str
    role: str = ""
    shape: str = ""
    provided: bool = False
    value_repr: str = ""


@dataclass(slots=True)
class ProblemAnalysis:
    source: str = "llm"
    title: str = ""
    problem_family: str = "Unknown"
    objective_sense: str = "minimize"
    objective_expression: str = ""
    variables: list[VariableSpec] = field(default_factory=list)
    constraints: list[ConstraintSpec] = field(default_factory=list)
    is_convex: bool | None = None
    convexity_summary: str = ""
    convexity_reason: str = ""
    modeling_notes: str = ""
    result_interpretation: str = ""
    data_status: str = "missing"
    data_symbols: list[DataSymbol] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    synthesis_allowed: bool = False
    synthesis_notes: str = ""

    @property
    def can_auto_solve(self) -> bool:
        return self.is_convex is True


@dataclass(slots=True)
class CodeBundle:
    model_code: str = ""
    executable_code: str = ""
    notes: str = ""
    assumptions: list[str] = field(default_factory=list)
    uses_synthesized_data: bool = False


@dataclass(slots=True)
class ExecutionSummary:
    status: str = "Not run"
    solver_name: str = ""
    optimal_value: str = "N/A"
    variable_values: dict[str, str] = field(default_factory=dict)
    dual_values: dict[str, str] = field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    workspace: str = ""
    duration_seconds: float = 0.0
    result_note: str = ""


@dataclass(slots=True)
class BuiltinExample:
    key: str
    title: str
    summary: str
    input_text: str
    analysis: ProblemAnalysis
    model_code: str


@dataclass(slots=True)
class WorkflowState:
    input_text: str = ""
    example_key: str | None = None
    analysis: ProblemAnalysis | None = None
    code: CodeBundle | None = None
    execution: ExecutionSummary | None = None
    logs: list[str] = field(default_factory=list)
