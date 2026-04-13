from __future__ import annotations

from convexopt_tutor_agent.core.schema import (
    BuiltinExample,
    ConstraintSpec,
    DataSymbol,
    ProblemAnalysis,
    VariableSpec,
)


def _analysis(
    *,
    title: str,
    family: str,
    objective_sense: str,
    objective_expression: str,
    variables: list[VariableSpec],
    constraints: list[ConstraintSpec],
    convexity_summary: str,
    convexity_reason: str,
    modeling_notes: str,
    result_interpretation: str,
    data_symbols: list[DataSymbol],
) -> ProblemAnalysis:
    return ProblemAnalysis(
        source="builtin",
        title=title,
        problem_family=family,
        objective_sense=objective_sense,
        objective_expression=objective_expression,
        variables=variables,
        constraints=constraints,
        is_convex=True,
        convexity_summary=convexity_summary,
        convexity_reason=convexity_reason,
        modeling_notes=modeling_notes,
        result_interpretation=result_interpretation,
        data_status="complete",
        data_symbols=data_symbols,
        assumptions=[],
        synthesis_allowed=False,
        synthesis_notes="This built-in example already includes numeric data.",
    )


def load_builtin_examples() -> list[BuiltinExample]:
    return [
        BuiltinExample(
            key="least_squares",
            title="Least Squares",
            summary="Least squares: minimize the squared residual norm.",
            input_text="Given a matrix A and a vector b, solve for x such that minimize ||Ax - b||_2^2.",
            analysis=_analysis(
                title="Least Squares",
                family="Least Squares",
                objective_sense="minimize",
                objective_expression="||Ax - b||_2^2",
                variables=[VariableSpec(name="x", shape="R^2", description="regression coefficients")],
                constraints=[],
                convexity_summary="This is a standard convex least-squares problem.",
                convexity_reason="The squared L2 norm of an affine expression is convex, and there are no additional non-convex constraints.",
                modeling_notes="Use `cp.sum_squares(A @ x - b)` for the residual sum of squares.",
                result_interpretation="The optimal solution gives the linear-model coefficients that best fit the observed data.",
                data_symbols=[
                    DataSymbol(name="A", role="matrix", shape="3x2", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="3", provided=True, value_repr="builtin"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
b = np.array([1.0, 0.0, 1.0])

x = cp.Variable(2, name="x")
objective = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = []
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {}
data_summary = "Built-in least-squares numeric data."
result_interpretation = "x.value contains the regression coefficients from the least-squares fit."
""",
        ),
        BuiltinExample(
            key="ridge",
            title="Ridge Regression",
            summary="Ridge regression: least squares with L2 regularization.",
            input_text="Given A, b, and lambda, solve minimize ||Ax - b||_2^2 + lambda * ||x||_2^2.",
            analysis=_analysis(
                title="Ridge Regression",
                family="Ridge Regression",
                objective_sense="minimize",
                objective_expression="||Ax - b||_2^2 + lambda ||x||_2^2",
                variables=[VariableSpec(name="x", shape="R^2", description="regression coefficients")],
                constraints=[ConstraintSpec(label="lambda_nonnegative", expression="lambda >= 0", kind="parameter")],
                convexity_summary="This is a standard convex ridge-regression problem.",
                convexity_reason="Both the squared residual term and the L2 regularizer are convex, and a nonnegative weighted sum preserves convexity.",
                modeling_notes="Add `lam * cp.sum_squares(x)` to the least-squares term.",
                result_interpretation="The optimal solution is typically smoother and has a smaller norm than ordinary least squares.",
                data_symbols=[
                    DataSymbol(name="A", role="matrix", shape="3x2", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="lambda", role="scalar", shape="1", provided=True, value_repr="0.5"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
b = np.array([1.0, 2.0, 2.5])
lam = 0.5

x = cp.Variable(2, name="x")
objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.sum_squares(x))
constraints = []
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {}
data_summary = "Built-in ridge-regression data with lambda = 0.5."
result_interpretation = "x.value contains the regression coefficients with L2 regularization."
""",
        ),
        BuiltinExample(
            key="lasso",
            title="LASSO",
            summary="LASSO: least squares with L1 regularization.",
            input_text="Given A, b, and lambda, solve minimize ||Ax - b||_2^2 + lambda * ||x||_1.",
            analysis=_analysis(
                title="LASSO",
                family="LASSO",
                objective_sense="minimize",
                objective_expression="||Ax - b||_2^2 + lambda ||x||_1",
                variables=[VariableSpec(name="x", shape="R^3", description="sparse regression coefficients")],
                constraints=[ConstraintSpec(label="lambda_nonnegative", expression="lambda >= 0", kind="parameter")],
                convexity_summary="This is a standard convex LASSO problem.",
                convexity_reason="The squared residual is convex, the L1 norm is also convex, and their sum remains convex.",
                modeling_notes="L1 regularization is commonly used to illustrate sparse solutions and feature selection.",
                result_interpretation="The optimal solution often drives some components to zero or near zero.",
                data_symbols=[
                    DataSymbol(name="A", role="matrix", shape="3x3", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="lambda", role="scalar", shape="1", provided=True, value_repr="0.3"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
b = np.array([1.0, 1.0, 0.5])
lam = 0.3

x = cp.Variable(3, name="x")
objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.norm1(x))
constraints = []
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {}
data_summary = "Built-in LASSO data with lambda = 0.3."
result_interpretation = "Near-zero entries in x.value show the sparsity encouraged by the L1 regularizer."
""",
        ),
        BuiltinExample(
            key="lp",
            title="Linear Program",
            summary="Linear program: linear objective with linear constraints.",
            input_text="Solve minimize c^T x subject to Ax <= b and x >= 0.",
            analysis=_analysis(
                title="Linear Program",
                family="LP",
                objective_sense="minimize",
                objective_expression="c^T x",
                variables=[VariableSpec(name="x", shape="R^3", domain="nonnegative", attributes=["nonnegative"])],
                constraints=[
                    ConstraintSpec(label="ineq", expression="Ax <= b", kind="linear_inequality"),
                    ConstraintSpec(label="nonnegative", expression="x >= 0", kind="linear_inequality"),
                ],
                convexity_summary="This is a standard convex linear program.",
                convexity_reason="The objective is linear, the constraints are linear, and the feasible set is a convex polyhedron.",
                modeling_notes="In CVXPY, `cp.Variable(nonneg=True)` is a convenient way to define a nonnegative variable.",
                result_interpretation="The optimal solution gives the minimum-cost decision vector subject to the constraints.",
                data_symbols=[
                    DataSymbol(name="c", role="vector", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="A", role="matrix", shape="2x3", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="2", provided=True, value_repr="builtin"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

c = np.array([1.0, 2.0, 0.5])
A = np.array([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
b = np.array([3.0, 4.0])

x = cp.Variable(3, nonneg=True, name="x")
constraints = [A @ x <= b]
objective = cp.Minimize(c @ x)
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {"Ax<=b": constraints[0]}
data_summary = "Built-in linear-program data."
result_interpretation = "x.value is the optimal resource allocation, and the dual values reflect marginal constraint costs."
""",
        ),
        BuiltinExample(
            key="qp",
            title="Quadratic Program",
            summary="Quadratic program: PSD quadratic objective with linear constraints.",
            input_text="Solve minimize 0.5 x^T Q x + q^T x subject to Ax <= b.",
            analysis=_analysis(
                title="Quadratic Program",
                family="QP",
                objective_sense="minimize",
                objective_expression="0.5 x^T Q x + q^T x",
                variables=[VariableSpec(name="x", shape="R^2", description="decision vector")],
                constraints=[ConstraintSpec(label="ineq", expression="Ax <= b", kind="linear_inequality")],
                convexity_summary="This is a standard convex quadratic program.",
                convexity_reason="The quadratic term is convex when Q is positive semidefinite, and the linear constraints preserve convexity.",
                modeling_notes="Use `cp.quad_form(x, Q)` for the quadratic term in CVXPY.",
                result_interpretation="The optimal solution balances quadratic cost against a linear bias term.",
                data_symbols=[
                    DataSymbol(name="Q", role="matrix", shape="2x2", provided=True, value_repr="builtin PSD"),
                    DataSymbol(name="q", role="vector", shape="2", provided=True, value_repr="builtin"),
                    DataSymbol(name="A", role="matrix", shape="2x2", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="2", provided=True, value_repr="builtin"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

Q = np.array([[2.0, 0.0], [0.0, 4.0]])
q = np.array([-2.0, -6.0])
A = np.array([[1.0, 1.0], [-1.0, 2.0]])
b = np.array([3.0, 2.0])

x = cp.Variable(2, name="x")
constraints = [A @ x <= b]
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + q @ x)
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {"Ax<=b": constraints[0]}
data_summary = "Built-in QP data with a positive-semidefinite Q matrix."
result_interpretation = "x.value is the optimal quadratic-program solution over the linear-feasible set."
""",
        ),
        BuiltinExample(
            key="portfolio",
            title="Portfolio Optimization",
            summary="Mean-variance portfolio optimization.",
            input_text="Solve minimize w^T Sigma w - gamma * mu^T w subject to sum(w) = 1 and w >= 0.",
            analysis=_analysis(
                title="Portfolio Optimization",
                family="Portfolio Optimization",
                objective_sense="minimize",
                objective_expression="w^T Sigma w - gamma * mu^T w",
                variables=[VariableSpec(name="w", shape="R^2", domain="nonnegative", attributes=["nonnegative"])],
                constraints=[
                    ConstraintSpec(label="budget", expression="sum(w) = 1", kind="equality"),
                    ConstraintSpec(label="nonnegative", expression="w >= 0", kind="linear_inequality"),
                ],
                convexity_summary="This is a standard convex portfolio-optimization problem.",
                convexity_reason="If the covariance matrix is positive semidefinite, the risk term is convex; the return term is linear and does not break convexity.",
                modeling_notes="Model the risk term with `cp.quad_form(w, Sigma)`.",
                result_interpretation="w.value gives the optimal asset-allocation weights.",
                data_symbols=[
                    DataSymbol(name="Sigma", role="covariance matrix", shape="2x2", provided=True, value_repr="builtin PSD"),
                    DataSymbol(name="mu", role="return vector", shape="2", provided=True, value_repr="builtin"),
                    DataSymbol(name="gamma", role="scalar", shape="1", provided=True, value_repr="0.5"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

Sigma = np.array([[0.10, 0.02], [0.02, 0.08]])
mu = np.array([0.12, 0.10])
gamma = 0.5

w = cp.Variable(2, nonneg=True, name="w")
constraints = [cp.sum(w) == 1]
objective = cp.Minimize(cp.quad_form(w, Sigma) - gamma * (mu @ w))
problem = cp.Problem(objective, constraints)

tracked_variables = {"w": w}
tracked_constraints = {"sum(w)=1": constraints[0]}
data_summary = "Built-in two-asset mean-variance data."
result_interpretation = "w.value gives the optimal allocation weights for the two assets."
""",
        ),
        BuiltinExample(
            key="logistic_regression",
            title="Logistic Regression",
            summary="Empirical-risk minimization for logistic regression.",
            input_text="Solve minimize sum(log(1 + exp(-y_i(a_i^T x + b)))) + lambda * ||x||_2^2.",
            analysis=_analysis(
                title="Logistic Regression",
                family="Logistic Regression",
                objective_sense="minimize",
                objective_expression="sum(log(1 + exp(-y_i(a_i^T x + b)))) + lambda ||x||_2^2",
                variables=[
                    VariableSpec(name="x", shape="R^2", description="classification weights"),
                    VariableSpec(name="b", shape="scalar", description="bias term"),
                ],
                constraints=[ConstraintSpec(label="lambda_nonnegative", expression="lambda >= 0", kind="parameter")],
                convexity_summary="This is a standard convex logistic-regression problem.",
                convexity_reason="The logistic loss is convex in an affine argument, and adding L2 regularization preserves convexity.",
                modeling_notes="In CVXPY, use `cp.logistic` to build the logistic-loss term.",
                result_interpretation="x.value and b.value define the separating hyperplane.",
                data_symbols=[
                    DataSymbol(name="A", role="feature matrix", shape="3x2", provided=True, value_repr="builtin"),
                    DataSymbol(name="y", role="label vector", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="lambda", role="scalar", shape="1", provided=True, value_repr="0.1"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[1.0, 2.0], [1.5, -0.5], [-1.0, 1.0]])
y = np.array([1.0, 1.0, -1.0])
lam = 0.1

x = cp.Variable(2, name="x")
b = cp.Variable(name="b")
loss = cp.sum(cp.logistic(-cp.multiply(y, A @ x + b)))
objective = cp.Minimize(loss + lam * cp.sum_squares(x))
constraints = []
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x, "b": b}
tracked_constraints = {}
data_summary = "Built-in binary logistic-regression data."
result_interpretation = "x.value and b.value determine the classification boundary."
""",
        ),
        BuiltinExample(
            key="svm",
            title="Soft-Margin SVM",
            summary="Soft-margin support vector machine.",
            input_text="Solve minimize 0.5 ||w||_2^2 + C * sum(xi_i), subject to y_i(a_i^T w + b) >= 1 - xi_i and xi_i >= 0.",
            analysis=_analysis(
                title="Soft-Margin SVM",
                family="SVM",
                objective_sense="minimize",
                objective_expression="0.5 ||w||_2^2 + C sum(xi_i)",
                variables=[
                    VariableSpec(name="w", shape="R^2", description="classification weights"),
                    VariableSpec(name="b", shape="scalar", description="bias term"),
                    VariableSpec(name="xi", shape="R^3", domain="nonnegative", attributes=["nonnegative"]),
                ],
                constraints=[
                    ConstraintSpec(label="margin", expression="y_i(a_i^T w + b) >= 1 - xi_i", kind="linear_inequality"),
                    ConstraintSpec(label="slack", expression="xi_i >= 0", kind="linear_inequality"),
                ],
                convexity_summary="This is a standard convex soft-margin SVM problem.",
                convexity_reason="The objective combines a convex quadratic term with a linear term, and the constraints are affine inequalities.",
                modeling_notes="`xi` is the slack variable that allows limited margin violations.",
                result_interpretation="w.value and b.value define the separating hyperplane, while xi.value measures the margin violations.",
                data_symbols=[
                    DataSymbol(name="A", role="feature matrix", shape="3x2", provided=True, value_repr="builtin"),
                    DataSymbol(name="y", role="label vector", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="C", role="scalar", shape="1", provided=True, value_repr="1.0"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[2.0, 1.0], [1.0, -1.0], [-1.5, -1.0]])
y = np.array([1.0, 1.0, -1.0])
C = 1.0

w = cp.Variable(2, name="w")
b = cp.Variable(name="b")
xi = cp.Variable(3, nonneg=True, name="xi")
constraints = [cp.multiply(y, A @ w + b) >= 1 - xi]
objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
problem = cp.Problem(objective, constraints)

tracked_variables = {"w": w, "b": b, "xi": xi}
tracked_constraints = {"margin": constraints[0]}
data_summary = "Built-in soft-margin SVM data."
result_interpretation = "w.value and b.value define the separating hyperplane, and xi.value shows the amount of margin violation."
""",
        ),
        BuiltinExample(
            key="basis_pursuit",
            title="Basis Pursuit",
            summary="Basis pursuit: minimize the L1 norm under equality constraints.",
            input_text="Solve minimize ||x||_1 subject to Ax = b.",
            analysis=_analysis(
                title="Basis Pursuit",
                family="Basis Pursuit",
                objective_sense="minimize",
                objective_expression="||x||_1",
                variables=[VariableSpec(name="x", shape="R^3", description="sparse representation coefficients")],
                constraints=[ConstraintSpec(label="equality", expression="Ax = b", kind="equality")],
                convexity_summary="This is a standard convex basis-pursuit problem.",
                convexity_reason="The L1 norm is convex, and equality constraints are affine, so the full problem is convex.",
                modeling_notes="The L1 norm is commonly used to encourage sparse solutions.",
                result_interpretation="The optimal solution typically favors a sparser representation vector.",
                data_symbols=[
                    DataSymbol(name="A", role="matrix", shape="2x3", provided=True, value_repr="builtin"),
                    DataSymbol(name="b", role="vector", shape="2", provided=True, value_repr="builtin"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

A = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
b = np.array([1.0, 0.5])

x = cp.Variable(3, name="x")
constraints = [A @ x == b]
objective = cp.Minimize(cp.norm1(x))
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {"Ax=b": constraints[0]}
data_summary = "Built-in basis-pursuit data."
result_interpretation = "x.value is the minimum-L1-norm solution satisfying Ax = b."
""",
        ),
        BuiltinExample(
            key="resource_allocation",
            title="Resource Allocation",
            summary="Resource allocation: maximize concave utility under a budget constraint.",
            input_text="Solve maximize sum(a_i * log(1 + x_i)) subject to sum(x_i) <= B and x_i >= 0.",
            analysis=_analysis(
                title="Resource Allocation",
                family="Resource Allocation",
                objective_sense="maximize",
                objective_expression="sum(a_i log(1 + x_i))",
                variables=[VariableSpec(name="x", shape="R^3", domain="nonnegative", attributes=["nonnegative"])],
                constraints=[
                    ConstraintSpec(label="budget", expression="sum(x_i) <= B", kind="linear_inequality"),
                    ConstraintSpec(label="nonnegative", expression="x_i >= 0", kind="linear_inequality"),
                ],
                convexity_summary="This is a standard convex resource-allocation problem.",
                convexity_reason="Log utility is concave, and maximizing a concave objective over a convex feasible set is a standard convex-optimization form.",
                modeling_notes="`cp.log1p(x)` directly represents log(1 + x).",
                result_interpretation="x.value gives the optimal resource allocation under the budget constraint.",
                data_symbols=[
                    DataSymbol(name="a", role="utility weights", shape="3", provided=True, value_repr="builtin"),
                    DataSymbol(name="B", role="budget", shape="1", provided=True, value_repr="4.0"),
                ],
            ),
            model_code="""import cvxpy as cp
import numpy as np

a = np.array([2.0, 1.0, 3.0])
B = 4.0

x = cp.Variable(3, nonneg=True, name="x")
constraints = [cp.sum(x) <= B]
objective = cp.Maximize(cp.sum(cp.multiply(a, cp.log1p(x))))
problem = cp.Problem(objective, constraints)

tracked_variables = {"x": x}
tracked_constraints = {"sum(x)<=B": constraints[0]}
data_summary = "Built-in resource-allocation data with budget B = 4."
result_interpretation = "x.value gives the optimal share of resources assigned to each project."
""",
        ),
    ]
