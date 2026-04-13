from __future__ import annotations

import json
from dataclasses import asdict

from convexopt_tutor_agent.core.schema import AppSettings, ProblemAnalysis


def _reasoning_line(settings: AppSettings) -> str:
    if settings.reasoning_mode == "deep":
        return "Reason carefully, but keep the final explanation concise and action-oriented."
    if settings.reasoning_mode == "balanced":
        return "Balance precision with concise explanations."
    return "Respond directly and concisely."


def build_analysis_prompts(problem_text: str, settings: AppSettings) -> tuple[str, str]:
    system_prompt = (
        "You are ConvexOpt Agent, an assistant for optimization problem solving. "
        "Your task is to parse optimization problems from natural language or math-like text. "
        "Be conservative: if convexity is unclear, set is_convex to null. "
        "Explain in concise technical language. "
        "Write every textual field in English, even if the input problem is written in another language. "
        "Return only one JSON object and no markdown. "
        f"{_reasoning_line(settings)}"
    )

    user_payload = {
        "task": "Parse the optimization problem and decide whether it is convex.",
        "requirements": {
            "language": "English",
            "schema": {
                "title": "short title for the problem",
                "problem_family": "LP/QP/SOCP/SDP/LASSO/Least Squares/etc.",
                "objective": {
                    "sense": "minimize|maximize",
                    "expression": "objective in math-like text",
                },
                "variables": [
                    {
                        "name": "variable name",
                        "shape": "scalar/vector/matrix shape",
                        "domain": "real/nonnegative/symmetric/etc.",
                        "attributes": ["list of attributes"],
                        "description": "short description",
                    }
                ],
                "constraints": [
                    {
                        "label": "c1",
                        "expression": "constraint expression",
                        "kind": "linear_inequality/equality/norm/soc/sdp/other",
                        "explanation": "short explanation",
                    }
                ],
                "is_convex": "true/false/null",
                "convexity_summary": "one-sentence verdict",
                "convexity_reason": "concise justification",
                "modeling_notes": "brief summary of the modeling choices",
                "result_interpretation": "brief summary of what the solution means",
                "data_status": "complete|symbolic|missing",
                "data_symbols": [
                    {
                        "name": "A",
                        "role": "matrix/vector/scalar parameter",
                        "shape": "shape if known",
                        "provided": "true/false",
                        "value_repr": "value if explicitly provided, otherwise 'not provided'",
                    }
                ],
                "assumptions": ["list any assumptions you make"],
                "synthesis_allowed": "true if small deterministic numeric data can be synthesized safely",
                "synthesis_notes": "explain how to synthesize deterministic data if needed",
            },
        },
        "problem_text": problem_text,
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False)


def build_code_prompts(
    problem_text: str,
    analysis: ProblemAnalysis,
    settings: AppSettings,
) -> tuple[str, str]:
    system_prompt = (
        "You generate safe CVXPY model-building code for an optimization-solving desktop app. "
        "Return only one JSON object and no markdown. "
        "The code must be deterministic, concise, and runnable on Windows. "
        "Write explanatory fields such as notes and assumptions in English. "
        "Do not import anything except cvxpy as cp, numpy as np, and math. "
        "Do not call problem.solve(). Do not print. Do not use file, network, subprocess, os, sys, pathlib, eval, exec, open, or importlib. "
        "The code must define these names: problem, tracked_variables, tracked_constraints, data_summary, result_interpretation. "
        "The name problem must refer to a cvxpy.Problem instance created by cp.Problem(objective, constraints); never reuse the name problem for numpy arrays, lists, dictionaries, descriptions, or raw data. "
        "tracked_variables and tracked_constraints should preferably be dictionaries that map readable names to CVXPY variables or constraints. "
        "If symbolic data is missing, synthesize a small deterministic instance and state that clearly. "
        f"{_reasoning_line(settings)}"
    )

    analysis_dict = asdict(analysis)
    user_payload = {
        "task": "Generate CVXPY model-building code.",
        "output_schema": {
            "should_generate_code": "true|false",
            "notes": "short explanation",
            "assumptions": ["list any code-generation assumptions"],
            "uses_synthesized_data": "true|false",
            "model_code": "plain Python code string without markdown fences",
        },
        "original_problem_text": problem_text,
        "analysis": analysis_dict,
    }
    return system_prompt, json.dumps(user_payload, ensure_ascii=False)
