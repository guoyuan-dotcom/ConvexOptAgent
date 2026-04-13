from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from convexopt_tutor_agent.core.schema import (
    AppSettings,
    BuiltinExample,
    ExecutionSummary,
    ProblemAnalysis,
    WorkflowState,
)
from convexopt_tutor_agent.core.settings_store import SettingsStore
from convexopt_tutor_agent.core.workflow import TutorWorkflowService
from convexopt_tutor_agent.ui.workers import TaskThread

WORKFLOW_TITLES = [
    "1. Problem Input",
    "2. Structured Parse",
    "3. Convexity Check",
    "4. Generate CVXPY Code",
    "5. Local Solve",
    "6. Solution Summary",
]


class MainWindow(QMainWindow):
    def __init__(
        self,
        examples: Iterable[BuiltinExample],
        workflow_service: TutorWorkflowService,
        settings_store: SettingsStore,
    ) -> None:
        super().__init__()
        self.examples = list(examples)
        self.examples_by_key = {example.key: example for example in self.examples}
        self.workflow_service = workflow_service
        self.settings_store = settings_store
        self.settings = self.settings_store.load()
        self.loaded_example_key: str | None = None
        self.current_workflow: WorkflowState | None = None
        self.analysis_thread: TaskThread | None = None
        self.execution_thread: TaskThread | None = None

        self.setWindowTitle("ConvexOpt Agent")
        self.resize(1650, 980)

        self._build_ui()
        self._load_settings_to_form()
        self._load_example_list()
        self._reset_outputs()
        self._refresh_provider_hint()
        self._append_log(
            "Application started. You can load a built-in example first, or enter a Moonshot API Key and analyze a free-text optimization problem."
        )

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        main_layout.addWidget(self._build_hero_card())
        main_layout.addWidget(self._build_workflow_card())

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_column())
        splitter.addWidget(self._build_center_column())
        splitter.addWidget(self._build_right_column())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 3)
        main_layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _build_hero_card(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("heroCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        title = QLabel("ConvexOpt Agent")
        title.setObjectName("heroTitle")
        subtitle = QLabel(
            "A desktop app for optimization problem solving: parse the problem, check convexity, generate CVXPY code, and solve it locally in a controlled environment."
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        return frame

    def _build_workflow_card(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("workflowCard")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        for title in WORKFLOW_TITLES:
            label = QLabel(title)
            label.setObjectName("stepLabel")
            layout.addWidget(label)
        return frame

    def _build_left_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_input_group(), 3)
        layout.addWidget(self._build_settings_group(), 2)
        layout.addWidget(self._build_workflow_status_group(), 2)
        return widget

    def _build_center_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_output_group("2. Structured Parse", "parse_output"), 2)
        layout.addWidget(self._build_output_group("3. Convexity Analysis", "convexity_output"), 2)

        code_group = self._build_output_group("4. CVXPY Code", "code_output", monospace=True)
        code_row = QHBoxLayout()
        code_row.addStretch(1)
        self.copy_code_button = QPushButton("Copy Code")
        self.copy_code_button.clicked.connect(self._copy_code)
        code_row.addWidget(self.copy_code_button)
        code_group.layout().addLayout(code_row)

        layout.addWidget(code_group, 3)
        return widget

    def _build_right_column(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        exec_group = QGroupBox("5. Solver Output")
        exec_layout = QVBoxLayout(exec_group)
        self.execution_output = self._make_output_editor(monospace=True)
        self.execute_button = QPushButton("Run Local Solve")
        self.execute_button.setObjectName("primaryButton")
        self.execute_button.setEnabled(False)
        self.execute_button.clicked.connect(self._execute_current_code)
        exec_layout.addWidget(self.execution_output)
        exec_layout.addWidget(self.execute_button)

        layout.addWidget(exec_group, 2)
        layout.addWidget(self._build_output_group("6. Solution Summary", "explanation_output"), 2)
        layout.addWidget(self._build_output_group("Workflow Log", "log_output", monospace=True), 2)
        return widget

    def _build_input_group(self) -> QWidget:
        group = QGroupBox("1. Problem Input")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        self.example_combo = QComboBox()
        self.example_combo.currentIndexChanged.connect(self._refresh_example_hint)
        self.load_example_button = QPushButton("Load Example")
        self.load_example_button.clicked.connect(self._load_selected_example)
        row.addWidget(self.example_combo, 1)
        row.addWidget(self.load_example_button)

        self.example_hint_label = QLabel(
            "Choose one of the 10 built-in examples, or enter a natural-language or math-like optimization problem."
        )
        self.example_hint_label.setWordWrap(True)

        self.input_editor = QPlainTextEdit()
        self.input_editor.setPlaceholderText("Example: minimize ||Ax - b||_2^2 subject to x >= 0")
        self.input_editor.textChanged.connect(self._sync_loaded_example_state)

        button_row = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze Problem")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self._analyze_current_input)

        self.clear_button = QPushButton("Clear")
        self.clear_button.setObjectName("warningButton")
        self.clear_button.clicked.connect(self._clear_all)

        button_row.addWidget(self.analyze_button)
        button_row.addWidget(self.clear_button)

        layout.addLayout(row)
        layout.addWidget(self.example_hint_label)
        layout.addWidget(self.input_editor, 1)
        layout.addLayout(button_row)
        return group

    def _build_settings_group(self) -> QWidget:
        group = QGroupBox("Settings")
        form = QFormLayout(group)

        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Moonshot API Key")

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://api.moonshot.cn/v1")

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("moonshot-v1-8k")
        self.model_edit.textChanged.connect(self._refresh_provider_hint)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)

        self.reasoning_mode_combo = QComboBox()
        self.reasoning_mode_combo.addItems(["default", "balanced", "deep"])

        self.request_timeout_spin = QSpinBox()
        self.request_timeout_spin.setRange(10, 300)

        self.execution_timeout_spin = QSpinBox()
        self.execution_timeout_spin.setRange(3, 300)

        self.provider_hint_label = QLabel()
        self.provider_hint_label.setWordWrap(True)

        self.save_settings_button = QPushButton("Save Settings")
        self.save_settings_button.clicked.connect(self._save_settings)

        form.addRow("API Key", self.api_key_edit)
        form.addRow("Base URL", self.base_url_edit)
        form.addRow("Model", self.model_edit)
        form.addRow("Temperature", self.temperature_spin)
        form.addRow("Reasoning", self.reasoning_mode_combo)
        form.addRow("LLM Timeout", self.request_timeout_spin)
        form.addRow("Execution Timeout", self.execution_timeout_spin)
        form.addRow("Provider", self.provider_hint_label)
        form.addRow("", self.save_settings_button)
        return group

    def _build_workflow_status_group(self) -> QWidget:
        group = QGroupBox("Workflow Status")
        layout = QVBoxLayout(group)

        self.workflow_list = QListWidget()
        for title in WORKFLOW_TITLES:
            self.workflow_list.addItem(QListWidgetItem(f"{title}  |  Waiting"))

        layout.addWidget(self.workflow_list)
        return group

    def _build_output_group(self, title: str, attr_name: str, monospace: bool = False) -> QWidget:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        editor = self._make_output_editor(monospace=monospace)
        setattr(self, attr_name, editor)
        layout.addWidget(editor)
        return group

    def _make_output_editor(self, monospace: bool = False) -> QPlainTextEdit:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        if monospace:
            editor.setStyleSheet(
                "QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; font-size: 12px; }"
            )
        return editor

    def _load_settings_to_form(self) -> None:
        self.api_key_edit.setText(self.settings.api_key)
        self.base_url_edit.setText(self.settings.base_url)
        self.model_edit.setText(self.settings.model)
        self.temperature_spin.setValue(self.settings.temperature)
        self.request_timeout_spin.setValue(self.settings.request_timeout_seconds)
        self.execution_timeout_spin.setValue(self.settings.execution_timeout_seconds)
        index = self.reasoning_mode_combo.findText(self.settings.reasoning_mode)
        if index >= 0:
            self.reasoning_mode_combo.setCurrentIndex(index)

    def _load_example_list(self) -> None:
        self.example_combo.clear()
        self.example_combo.addItem("Choose a built-in example...", "")
        for example in self.examples:
            self.example_combo.addItem(example.title, example.key)
        self._refresh_example_hint()

    def _refresh_example_hint(self) -> None:
        example_key = self.example_combo.currentData()
        if not example_key:
            self.example_hint_label.setText(
                "Choose one of the 10 built-in examples, or enter a natural-language or math-like optimization problem."
            )
            return
        example = self.examples_by_key[example_key]
        self.example_hint_label.setText(example.summary)

    def _refresh_provider_hint(self) -> None:
        model_name = self.model_edit.text().strip() or self.settings.model
        if self.workflow_service.kimi_client.supports_json_mode(model_name):
            self.provider_hint_label.setText("This model will use JSON mode for more stable structured output.")
        else:
            self.provider_hint_label.setText(
                "This model does not support JSON mode and will fall back to plain-text JSON extraction."
            )

    def _load_selected_example(self) -> None:
        example_key = self.example_combo.currentData()
        if not example_key:
            QMessageBox.information(self, "Notice", "Please choose a built-in example first.")
            return

        example = self.examples_by_key[example_key]
        self.loaded_example_key = example.key
        self.input_editor.blockSignals(True)
        self.input_editor.setPlainText(example.input_text)
        self.input_editor.blockSignals(False)
        self.current_workflow = None
        self.execute_button.setEnabled(False)
        self._append_log(f"Loaded example: {example.title}")
        self._set_step_status(0, "Example loaded")

    def _sync_loaded_example_state(self) -> None:
        if not self.loaded_example_key:
            return
        example = self.examples_by_key.get(self.loaded_example_key)
        if example and self.input_editor.toPlainText() != example.input_text:
            self.loaded_example_key = None
            self._append_log("Input changed. Switching back to free-text mode.")

    def _collect_settings(self) -> AppSettings:
        return AppSettings(
            api_key=self.api_key_edit.text().strip(),
            base_url=self.base_url_edit.text().strip() or "https://api.moonshot.cn/v1",
            model=self.model_edit.text().strip() or "moonshot-v1-8k",
            temperature=float(self.temperature_spin.value()),
            reasoning_mode=self.reasoning_mode_combo.currentText(),
            request_timeout_seconds=int(self.request_timeout_spin.value()),
            execution_timeout_seconds=int(self.execution_timeout_spin.value()),
        )

    def _save_settings(self) -> None:
        self.settings = self._collect_settings()
        self.settings_store.save(self.settings)
        self._refresh_provider_hint()
        self._append_log("Settings saved.")
        self.statusBar().showMessage("Settings saved", 3000)

    def _analyze_current_input(self) -> None:
        input_text = self.input_editor.toPlainText().strip()
        if not input_text:
            QMessageBox.warning(self, "Empty Input", "Please enter an optimization problem, or load a built-in example first.")
            return

        self.settings = self._collect_settings()
        self.execute_button.setEnabled(False)
        self.current_workflow = None
        self._reset_outputs(keep_logs=False)
        self._append_log("Starting problem analysis.")
        self._set_buttons_busy(True)
        self._set_step_status(0, "Received")
        self._set_step_status(1, "Running")

        self.analysis_thread = TaskThread(
            self.workflow_service.analyze_problem,
            input_text,
            self.settings,
            self.loaded_example_key,
        )
        self.analysis_thread.progress.connect(self._handle_progress_message)
        self.analysis_thread.succeeded.connect(self._handle_analysis_success)
        self.analysis_thread.failed.connect(self._handle_analysis_failure)
        self.analysis_thread.start()

    def _execute_current_code(self) -> None:
        if self.current_workflow is None:
            QMessageBox.warning(self, "Cannot Execute", "There is no executable analysis result yet.")
            return
        self.settings = self._collect_settings()
        self._append_log("Starting local solve.")
        self._set_buttons_busy(True)
        self._set_step_status(4, "Running")

        self.execution_thread = TaskThread(
            self.workflow_service.execute_solution,
            self.current_workflow,
            self.settings,
        )
        self.execution_thread.progress.connect(self._handle_progress_message)
        self.execution_thread.succeeded.connect(self._handle_execution_success)
        self.execution_thread.failed.connect(self._handle_execution_failure)
        self.execution_thread.start()

    def _handle_progress_message(self, message: str) -> None:
        self._append_log(message)

    def _handle_analysis_success(self, workflow: WorkflowState) -> None:
        self.current_workflow = workflow
        self._render_analysis(workflow)
        self._set_buttons_busy(False)
        self.statusBar().showMessage("Analysis complete", 3000)

    def _handle_analysis_failure(self, message: str) -> None:
        self._set_buttons_busy(False)
        self._append_log(f"Analysis failed: {message}")
        QMessageBox.critical(self, "Analysis Failed", message)

    def _handle_execution_success(self, summary: ExecutionSummary) -> None:
        self._set_buttons_busy(False)
        self._set_step_status(4, "Done")
        if self.current_workflow is not None:
            self.current_workflow = replace(self.current_workflow, execution=summary)
            self.explanation_output.setPlainText(self._format_explanation(self.current_workflow))
        self.execution_output.setPlainText(self._format_execution(summary))
        self.statusBar().showMessage("Local solve complete", 3000)

    def _handle_execution_failure(self, message: str) -> None:
        self._set_buttons_busy(False)
        self._append_log(f"Execution failed: {message}")
        self._set_step_status(4, "Failed")
        QMessageBox.critical(self, "Execution Failed", message)

    def _render_analysis(self, workflow: WorkflowState) -> None:
        analysis = workflow.analysis
        if analysis is None:
            return

        self.parse_output.setPlainText(self._format_structure(analysis))
        self.convexity_output.setPlainText(self._format_convexity(analysis))
        self.explanation_output.setPlainText(self._format_explanation(workflow))
        self.execution_output.setPlainText(
            "Local solve has not been run yet.\n"
            "Click 'Run Local Solve' to execute the generated Python + CVXPY code inside a controlled temporary workspace."
        )

        self._set_step_status(1, "Done")
        self._set_step_status(2, "Done")
        self._set_step_status(5, "Done")

        if workflow.code is not None:
            self.code_output.setPlainText(workflow.code.executable_code)
            self._set_step_status(3, "Done")
            self._set_step_status(4, "Ready")
            self.execute_button.setEnabled(True)
        else:
            self.code_output.setPlainText(
                "# No executable code is available.\n# Possible reason: the problem was classified as non-convex, or convexity could not be confirmed."
            )
            self._set_step_status(3, "Not generated")
            if analysis.is_convex is False:
                self._set_step_status(4, "Stopped")
            else:
                self._set_step_status(4, "Needs review")
            self.execute_button.setEnabled(False)

    def _format_structure(self, analysis: ProblemAnalysis) -> str:
        variables = (
            "\n".join(
                f"- {item.name}: shape={item.shape}, domain={item.domain}, attrs={', '.join(item.attributes) or 'none'}, {item.description}"
                for item in analysis.variables
            )
            if analysis.variables
            else "- Not identified"
        )
        constraints = (
            "\n".join(
                f"- {item.label}: {item.expression} [{item.kind}] {item.explanation}".strip()
                for item in analysis.constraints
            )
            if analysis.constraints
            else "- No explicit constraints"
        )
        data_symbols = (
            "\n".join(
                f"- {item.name}: role={item.role}, shape={item.shape or 'unknown'}, provided={item.provided}, value={item.value_repr or 'N/A'}"
                for item in analysis.data_symbols
            )
            if analysis.data_symbols
            else "- No parameters listed"
        )
        assumptions = (
            "\n".join(f"- {item}" for item in analysis.assumptions)
            if analysis.assumptions
            else "- No additional assumptions"
        )
        return (
            f"Title: {analysis.title or 'Untitled problem'}\n"
            f"Problem family: {analysis.problem_family}\n"
            f"Objective: {analysis.objective_sense} {analysis.objective_expression}\n\n"
            f"Variables:\n{variables}\n\n"
            f"Constraints:\n{constraints}\n\n"
            f"Data status: {analysis.data_status}\n"
            f"Parameters:\n{data_symbols}\n\n"
            f"Assumptions:\n{assumptions}"
        )

    def _format_convexity(self, analysis: ProblemAnalysis) -> str:
        if analysis.is_convex is True:
            verdict = "Verdict: Convex"
        elif analysis.is_convex is False:
            verdict = "Verdict: Non-convex"
        else:
            verdict = "Verdict: Not yet confirmed"

        auto_solve = "Auto-solve allowed" if analysis.can_auto_solve else "Auto-solve disabled"
        return (
            f"{verdict}\n"
            f"Summary: {analysis.convexity_summary}\n\n"
            f"Reason:\n{analysis.convexity_reason}\n\n"
            f"Execution policy: {auto_solve}\n"
            f"Data synthesis note: {analysis.synthesis_notes or 'None'}"
        )

    def _format_explanation(self, workflow: WorkflowState) -> str:
        analysis = workflow.analysis
        if analysis is None:
            return "Waiting for analysis results."

        code_notes = workflow.code.notes if workflow.code is not None else "No code has been generated yet."
        execution_note = workflow.execution.result_note if workflow.execution is not None else "Execution has not been run yet."
        assumptions = (
            "\n".join(f"- {item}" for item in analysis.assumptions)
            if analysis.assumptions
            else "- No additional assumptions"
        )

        code_assumptions = (
            "\n".join(f"- {item}" for item in (workflow.code.assumptions if workflow.code else []))
            if workflow.code and workflow.code.assumptions
            else "- No code-generation assumptions"
        )

        return (
            "Modeling summary:\n"
            f"{analysis.modeling_notes}\n\n"
            "Convexity basis:\n"
            f"{analysis.convexity_reason}\n\n"
            "Result interpretation:\n"
            f"{analysis.result_interpretation}\n\n"
            "Analysis assumptions:\n"
            f"{assumptions}\n\n"
            "Code notes:\n"
            f"{code_notes}\n\n"
            "Code assumptions:\n"
            f"{code_assumptions}\n\n"
            "Execution summary:\n"
            f"{execution_note}"
        )

    def _format_execution(self, summary: ExecutionSummary) -> str:
        variables = (
            "\n".join(f"- {name}: {value}" for name, value in summary.variable_values.items())
            if summary.variable_values
            else "- None"
        )
        duals = (
            "\n".join(f"- {name}: {value}" for name, value in summary.dual_values.items())
            if summary.dual_values
            else "- None"
        )
        stdout_block = summary.stdout or "(empty)"
        stderr_block = summary.stderr or "(empty)"
        return (
            f"Status: {summary.status}\n"
            f"Solver: {summary.solver_name or 'N/A'}\n"
            f"Optimal value: {summary.optimal_value}\n"
            f"Elapsed time: {summary.duration_seconds:.3f} seconds\n"
            f"Workspace: {summary.workspace or 'N/A'}\n\n"
            f"Variable values:\n{variables}\n\n"
            f"Dual values:\n{duals}\n\n"
            f"Result note:\n{summary.result_note or 'None'}\n\n"
            f"stdout:\n{stdout_block}\n\n"
            f"stderr:\n{stderr_block}"
        )

    def _copy_code(self) -> None:
        code = self.code_output.toPlainText().strip()
        if not code:
            QMessageBox.information(self, "Notice", "There is no code to copy.")
            return
        QGuiApplication.clipboard().setText(code)
        self._append_log("Code copied to clipboard.")
        self.statusBar().showMessage("Code copied", 3000)

    def _clear_all(self) -> None:
        self.loaded_example_key = None
        self.current_workflow = None
        self.input_editor.clear()
        self._reset_outputs(keep_logs=False)
        self._append_log("Cleared input and outputs.")
        self.execute_button.setEnabled(False)

    def _reset_outputs(self, keep_logs: bool = True) -> None:
        self.parse_output.setPlainText("Waiting for problem analysis.")
        self.convexity_output.setPlainText("Waiting for analysis results.")
        self.code_output.setPlainText("# Waiting for generated CVXPY code")
        self.execution_output.setPlainText("Waiting for local execution.")
        self.explanation_output.setPlainText("Waiting for the solution summary.")
        if not keep_logs:
            self.log_output.clear()
        for index in range(len(WORKFLOW_TITLES)):
            self._set_step_status(index, "Waiting")

    def _set_buttons_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy)
        self.load_example_button.setEnabled(not busy)
        self.save_settings_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        if busy:
            self.execute_button.setEnabled(False)
        elif self.current_workflow is not None and self.current_workflow.code is not None:
            self.execute_button.setEnabled(True)

    def _set_step_status(self, index: int, status: str) -> None:
        item = self.workflow_list.item(index)
        if item is None:
            return
        item.setText(f"{WORKFLOW_TITLES[index]}  |  {status}")

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
