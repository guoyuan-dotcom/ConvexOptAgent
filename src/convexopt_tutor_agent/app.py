from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from convexopt_tutor_agent.core.settings_store import SettingsStore
from convexopt_tutor_agent.core.workflow import TutorWorkflowService
from convexopt_tutor_agent.execution.local_runner import LocalExecutionRunner
from convexopt_tutor_agent.examples.builtin_examples import load_builtin_examples
from convexopt_tutor_agent.llm.kimi_adapter import KimiClient
from convexopt_tutor_agent.ui.main_window import MainWindow


def apply_app_style(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background: #f4f7fb;
            color: #203047;
        }
        QGroupBox {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 12px;
            margin-top: 12px;
            font-weight: 600;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 6px;
            color: #214a72;
        }
        QLabel#heroTitle {
            font-size: 24px;
            font-weight: 700;
            color: #17395c;
        }
        QLabel#heroSubtitle {
            color: #52657d;
            font-size: 12px;
        }
        QFrame#heroCard, QFrame#workflowCard {
            background: #ffffff;
            border: 1px solid #dbe3ee;
            border-radius: 14px;
        }
        QLabel#stepLabel {
            background: #eef4fb;
            color: #214a72;
            border: 1px solid #d8e4f2;
            border-radius: 14px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox {
            background: #fbfdff;
            border: 1px solid #cad6e3;
            border-radius: 10px;
            padding: 6px 8px;
            selection-background-color: #9ec3ea;
        }
        QPlainTextEdit[readOnly="true"] {
            background: #f8fafc;
        }
        QListWidget {
            background: #fbfdff;
            border: 1px solid #cad6e3;
            border-radius: 10px;
            padding: 6px;
        }
        QPushButton {
            background: #eef3f8;
            border: 1px solid #c8d5e3;
            border-radius: 10px;
            padding: 7px 12px;
            color: #1d334d;
        }
        QPushButton:hover {
            background: #e4edf7;
        }
        QPushButton#primaryButton {
            background: #2c6aa5;
            color: #ffffff;
            border: 1px solid #255c8f;
            font-weight: 600;
        }
        QPushButton#primaryButton:hover {
            background: #255c8f;
        }
        QPushButton#warningButton {
            background: #fff4e5;
            border: 1px solid #f0d0a1;
        }
        QStatusBar {
            background: #ffffff;
            border-top: 1px solid #dbe3ee;
        }
        """
    )


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ConvexOpt Agent")
    app.setOrganizationName("ConvexOptAgent")
    apply_app_style(app)

    settings_store = SettingsStore()
    examples = load_builtin_examples()
    workflow_service = TutorWorkflowService(
        kimi_client=KimiClient(),
        execution_runner=LocalExecutionRunner(),
        examples=examples,
    )

    window = MainWindow(
        examples=examples,
        workflow_service=workflow_service,
        settings_store=settings_store,
    )
    window.show()
    return app.exec()
