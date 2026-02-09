"""
Inverse problem solver for parameter estimation.

Estimates (α, β, γ, δ, Γ) from observed powder diffraction pattern.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass, field

from .forward_model import PowderDiffractionModel

# Try to import scikit-optimize for Bayesian optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.callbacks import EarlyStopper
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False


@dataclass
class InverseResult:
    """Result of inverse problem solution."""
    alpha: float
    beta: float
    gamma: float
    delta: float
    Gamma: float
    success: bool
    loss: float
    n_iterations: int
    message: str

    @property
    def params(self) -> Tuple[float, float, float, float, float]:
        """Return parameters as tuple."""
        return (self.alpha, self.beta, self.gamma, self.delta, self.Gamma)


@dataclass
class BayesianInverseResult:
    """Result of Bayesian optimization inverse problem solution."""
    alpha: float
    beta: float
    gamma: float
    delta: float
    loss: float
    n_evaluations: int
    convergence_trace: List[float]
    gp_model: object = field(default=None, repr=False)

    @property
    def params(self) -> Tuple[float, float, float, float]:
        """Return parameters as tuple (no Gamma since it's fixed)."""
        return (self.alpha, self.beta, self.gamma, self.delta)


def logit(x: float) -> float:
    """Logit transform: [0,1] → ℝ."""
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return np.log(x / (1 - x))


def sigmoid(y: float) -> float:
    """Sigmoid transform: ℝ → [0,1]."""
    return 1 / (1 + np.exp(-y))


def softplus(x: float) -> float:
    """Softplus: ℝ → (0, ∞)."""
    return np.log1p(np.exp(x))


def softplus_inv(y: float) -> float:
    """Inverse softplus: (0, ∞) → ℝ."""
    return np.log(np.expm1(y))


def transform_params(alpha: float, beta: float, gamma: float,
                    delta: float, Gamma: float) -> np.ndarray:
    """Transform constrained parameters to unconstrained space."""
    return np.array([
        logit(alpha),
        logit(beta),
        logit(gamma),
        logit(delta),
        softplus_inv(Gamma)
    ])


def inverse_transform_params(theta: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Transform unconstrained parameters back to constrained space."""
    alpha = sigmoid(theta[0])
    beta = sigmoid(theta[1])
    gamma = sigmoid(theta[2])
    delta = sigmoid(theta[3])
    Gamma = softplus(theta[4])
    return alpha, beta, gamma, delta, Gamma


class InverseSolver:
    """
    Inverse problem solver for powder diffraction.

    Uses multi-start optimization with parameter reparameterization.
    """

    def __init__(self, model: PowderDiffractionModel,
                Q_grid: np.ndarray,
                I_obs: np.ndarray):
        """
        Initialize solver.

        Parameters
        ----------
        model : PowderDiffractionModel
            Forward model
        Q_grid : np.ndarray
            Q values
        I_obs : np.ndarray
            Observed intensities
        """
        self.model = model
        self.Q_grid = np.asarray(Q_grid)
        self.I_obs = np.asarray(I_obs)
        self._n_evals = 0

    def objective(self, theta: np.ndarray) -> float:
        """
        Compute least-squares objective in unconstrained space.

        L(θ) = Σ_i (I_obs(Q_i) - I_model(Q_i; θ))²
        """
        self._n_evals += 1

        alpha, beta, gamma, delta, Gamma = inverse_transform_params(theta)

        try:
            I_model = self.model.compute_pattern_fast(
                self.Q_grid, Gamma, alpha, beta, gamma, delta)
            residuals = self.I_obs - I_model
            return np.sum(residuals ** 2)
        except Exception:
            return 1e20  # Return large value on failure

    def objective_constrained(self, params: np.ndarray) -> float:
        """
        Compute objective in constrained space (for bounded optimizers).
        """
        self._n_evals += 1
        alpha, beta, gamma, delta, Gamma = params

        try:
            I_model = self.model.compute_pattern_fast(
                self.Q_grid, Gamma, alpha, beta, gamma, delta)
            residuals = self.I_obs - I_model
            return np.sum(residuals ** 2)
        except Exception:
            return 1e20

    def solve(self, method: str = 'L-BFGS-B',
             x0: Optional[np.ndarray] = None,
             n_restarts: int = 5,
             verbose: bool = True) -> InverseResult:
        """
        Solve inverse problem.

        Parameters
        ----------
        method : str
            Optimization method: 'L-BFGS-B', 'differential_evolution', or 'multi_start'
        x0 : np.ndarray, optional
            Initial guess [α, β, γ, δ, Γ]. Uses random if None.
        n_restarts : int
            Number of random restarts for multi_start method
        verbose : bool
            Print progress

        Returns
        -------
        InverseResult
            Optimization result
        """
        self._n_evals = 0

        if method == 'differential_evolution':
            return self._solve_de(verbose)
        elif method == 'multi_start':
            return self._solve_multi_start(n_restarts, verbose)
        else:
            return self._solve_local(x0, method, verbose)

    def _solve_local(self, x0: Optional[np.ndarray],
                    method: str, verbose: bool) -> InverseResult:
        """Solve with local optimizer."""
        if x0 is None:
            x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.02])

        # Use bounded optimization
        bounds = [(0.001, 0.999)] * 4 + [(0.001, 0.5)]

        if verbose:
            print(f"Starting local optimization with {method}...")
            print(f"  Initial: α={x0[0]:.3f}, β={x0[1]:.3f}, γ={x0[2]:.3f}, δ={x0[3]:.3f}, Γ={x0[4]:.4f}")

        result = minimize(
            self.objective_constrained,
            x0,
            method=method,
            bounds=bounds,
            options={'maxiter': 500, 'disp': verbose}
        )

        alpha, beta, gamma, delta, Gamma = result.x

        if verbose:
            print(f"  Final: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}, Γ={Gamma:.4f}")
            print(f"  Loss: {result.fun:.6e}")
            print(f"  Evaluations: {self._n_evals}")

        return InverseResult(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            Gamma=Gamma,
            success=result.success,
            loss=result.fun,
            n_iterations=self._n_evals,
            message=result.message if hasattr(result, 'message') else str(result)
        )

    def _solve_de(self, verbose: bool) -> InverseResult:
        """Solve with differential evolution (global optimizer)."""
        bounds = [(0.001, 0.999)] * 4 + [(0.001, 0.5)]

        if verbose:
            print("Starting differential evolution...")

        result = differential_evolution(
            self.objective_constrained,
            bounds,
            maxiter=100,
            tol=1e-6,
            workers=1,
            updating='deferred',
            disp=verbose
        )

        alpha, beta, gamma, delta, Gamma = result.x

        if verbose:
            print(f"  Final: α={alpha:.3f}, β={beta:.3f}, γ={gamma:.3f}, δ={delta:.3f}, Γ={Gamma:.4f}")
            print(f"  Loss: {result.fun:.6e}")
            print(f"  Evaluations: {self._n_evals}")

        return InverseResult(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            Gamma=Gamma,
            success=result.success,
            loss=result.fun,
            n_iterations=self._n_evals,
            message=result.message
        )

    def _solve_multi_start(self, n_restarts: int, verbose: bool) -> InverseResult:
        """Solve with multiple random restarts."""
        rng = np.random.default_rng(42)
        best_result = None
        best_loss = np.inf

        for i in range(n_restarts):
            # Random initial point
            x0 = np.array([
                rng.uniform(0.1, 0.9),
                rng.uniform(0.1, 0.9),
                rng.uniform(0.1, 0.9),
                rng.uniform(0.1, 0.9),
                rng.uniform(0.01, 0.1)
            ])

            if verbose:
                print(f"\nRestart {i+1}/{n_restarts}:")

            result = self._solve_local(x0, 'L-BFGS-B', verbose=False)

            if verbose:
                print(f"  Loss: {result.loss:.6e}")

            if result.loss < best_loss:
                best_loss = result.loss
                best_result = result

        if verbose:
            print(f"\nBest result:")
            print(f"  α={best_result.alpha:.3f}, β={best_result.beta:.3f}, "
                  f"γ={best_result.gamma:.3f}, δ={best_result.delta:.3f}, "
                  f"Γ={best_result.Gamma:.4f}")
            print(f"  Loss: {best_result.loss:.6e}")

        return best_result


class DeltaYStopper:
    """Early stopping callback for Bayesian optimization."""

    def __init__(self, delta: float = 1e-6, n_best: int = 30):
        """
        Stop if the best objective hasn't improved by delta in n_best iterations.

        Parameters
        ----------
        delta : float
            Minimum relative improvement threshold (as fraction of best value)
        n_best : int
            Number of iterations to check for improvement
        """
        self.delta = delta
        self.n_best = n_best
        self._best_values = []

    def __call__(self, result):
        """Check stopping criterion."""
        self._best_values.append(result.fun)

        if len(self._best_values) < self.n_best:
            return False

        # Check relative improvement in the last n_best iterations
        recent = self._best_values[-self.n_best:]
        best_recent = min(recent)

        # Use relative improvement to handle varying scales
        if best_recent > 0:
            relative_improvement = (recent[0] - best_recent) / best_recent
        else:
            relative_improvement = recent[0] - best_recent

        return relative_improvement < self.delta


class BayesianInverseSolver:
    """
    Bayesian optimization inverse solver for powder diffraction.

    Uses Gaussian Process-based Bayesian optimization to estimate
    (α, β, γ, δ) with fixed instrument broadening Γ.

    This approach is sample-efficient and handles nonconvex objective
    landscapes well, making it suitable for the expensive forward model.
    """

    def __init__(self, model: PowderDiffractionModel,
                 Q_grid: np.ndarray,
                 I_obs: np.ndarray,
                 Gamma_fixed: float):
        """
        Initialize Bayesian optimization solver.

        Parameters
        ----------
        model : PowderDiffractionModel
            Forward model for computing diffraction patterns
        Q_grid : np.ndarray
            Q values (Å⁻¹) at which intensities are observed
        I_obs : np.ndarray
            Observed intensities
        Gamma_fixed : float
            Fixed instrument broadening parameter (not estimated)
        """
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )

        self.model = model
        self.Q_grid = np.asarray(Q_grid)
        self.I_obs = np.asarray(I_obs)
        self.Gamma_fixed = Gamma_fixed
        self._n_evals = 0
        self._eval_trace = []

    def objective(self, params: List[float]) -> float:
        """
        Compute sum of squared errors objective.

        Parameters
        ----------
        params : list
            [alpha, beta, gamma, delta] parameters

        Returns
        -------
        float
            Sum of squared errors between observed and model intensities
        """
        self._n_evals += 1
        alpha, beta, gamma, delta = params

        try:
            I_model = self.model.compute_pattern_fast(
                self.Q_grid, self.Gamma_fixed, alpha, beta, gamma, delta)
            sse = np.sum((self.I_obs - I_model) ** 2)
            self._eval_trace.append(sse)
            return sse
        except Exception as e:
            # Return large value on failure
            large_val = 1e20
            self._eval_trace.append(large_val)
            return large_val

    def solve(self, n_calls: int = 100, n_initial: int = 10,
              verbose: bool = True, random_state: int = 42) -> BayesianInverseResult:
        """
        Run Bayesian optimization to find optimal parameters.

        Parameters
        ----------
        n_calls : int
            Total number of function evaluations (default 100)
        n_initial : int
            Initial random samples before fitting GP (default 10)
        verbose : bool
            Print progress information
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        BayesianInverseResult
            Optimization result containing best parameters and diagnostics
        """
        self._n_evals = 0
        self._eval_trace = []

        # Define search space - slightly inside [0,1] to avoid boundary issues
        space = [
            Real(0.01, 0.99, name='alpha'),
            Real(0.01, 0.99, name='beta'),
            Real(0.01, 0.99, name='gamma'),
            Real(0.01, 0.99, name='delta'),
        ]

        if verbose:
            print("Starting Bayesian Optimization...")
            print(f"  Fixed Γ = {self.Gamma_fixed}")
            print(f"  n_calls = {n_calls}, n_initial = {n_initial}")

        # Early stopping callback - use None to disable (let n_calls control)
        early_stopper = None  # Disabled - let BO run full n_calls

        # Callback for verbose output
        def verbose_callback(result):
            if verbose and len(result.func_vals) % 10 == 0:
                current_best = min(result.func_vals)
                print(f"  Iteration {len(result.func_vals)}: best loss = {current_best:.6e}")
            return False  # Never early stop from verbose callback

        # Run Bayesian optimization
        result = gp_minimize(
            func=self.objective,
            dimensions=space,
            n_calls=n_calls,
            n_initial_points=n_initial,
            acq_func='EI',  # Expected Improvement
            noise=1e-10,    # Near-deterministic function
            random_state=random_state,
            callback=verbose_callback if verbose else None,
        )

        # Extract best parameters
        alpha_best, beta_best, gamma_best, delta_best = result.x

        # Build convergence trace (best-so-far at each iteration)
        convergence_trace = []
        best_so_far = float('inf')
        for val in result.func_vals:
            best_so_far = min(best_so_far, val)
            convergence_trace.append(best_so_far)

        if verbose:
            print(f"\nOptimization complete!")
            print(f"  Evaluations: {len(result.func_vals)}")
            print(f"  Best loss: {result.fun:.6e}")
            print(f"  Best params: α={alpha_best:.4f}, β={beta_best:.4f}, "
                  f"γ={gamma_best:.4f}, δ={delta_best:.4f}")

        return BayesianInverseResult(
            alpha=alpha_best,
            beta=beta_best,
            gamma=gamma_best,
            delta=delta_best,
            loss=result.fun,
            n_evaluations=len(result.func_vals),
            convergence_trace=convergence_trace,
            gp_model=result.models[-1] if result.models else None,
        )


def validate_bo_solver(true_params: Tuple[float, float, float, float] = (0.7, 0.4, 0.6, 0.5),
                       Gamma: float = 0.02,
                       n_calls: int = 100,
                       N_a: int = 5, N_b: int = 5, N_c: int = 10,
                       n_Q: int = 50, verbose: bool = True) -> dict:
    """
    Validate Bayesian optimization solver with synthetic data.

    Parameters
    ----------
    true_params : tuple
        True (α, β, γ, δ) parameters
    Gamma : float
        Fixed instrument broadening
    n_calls : int
        Number of BO iterations
    N_a, N_b, N_c : int
        Crystallite dimensions (smaller for speed)
    n_Q : int
        Number of Q points
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Validation results including errors and timing
    """
    import time

    true_alpha, true_beta, true_gamma, true_delta = true_params

    # Create forward model
    model = PowderDiffractionModel(N_a=N_a, N_b=N_b, N_c=N_c)
    Q_grid = np.linspace(0.5, 4.0, n_Q)

    # Generate synthetic data
    if verbose:
        print("Generating synthetic data...")
        print(f"  True params: α={true_alpha}, β={true_beta}, γ={true_gamma}, δ={true_delta}")
        print(f"  Fixed Γ={Gamma}")

    I_obs = model.compute_pattern_fast(
        Q_grid, Gamma, true_alpha, true_beta, true_gamma, true_delta)

    # Solve inverse problem with Bayesian optimization
    solver = BayesianInverseSolver(model, Q_grid, I_obs, Gamma_fixed=Gamma)

    t0 = time.time()
    result = solver.solve(n_calls=n_calls, verbose=verbose)
    elapsed = time.time() - t0

    # Compare
    errors = {
        'alpha': abs(result.alpha - true_alpha),
        'beta': abs(result.beta - true_beta),
        'gamma': abs(result.gamma - true_gamma),
        'delta': abs(result.delta - true_delta),
    }

    if verbose:
        print("\nParameter errors:")
        for name, err in errors.items():
            print(f"  {name}: {err:.4f}")
        print(f"\nTotal time: {elapsed:.1f} s")

    max_error = max(errors.values())
    success = max_error < 0.05

    if verbose:
        if success:
            print("PASSED: Parameters recovered within tolerance (< 0.05)")
        else:
            print(f"WARNING: Max error {max_error:.4f} exceeds tolerance 0.05")

    return {
        'true_params': true_params,
        'estimated_params': result.params,
        'errors': errors,
        'max_error': max_error,
        'loss': result.loss,
        'n_evaluations': result.n_evaluations,
        'elapsed_time': elapsed,
        'convergence_trace': result.convergence_trace,
        'success': success,
    }


def validate_inverse_solver(true_params: Tuple[float, float, float, float, float] = None,
                           N_a: int = 5, N_b: int = 5, N_c: int = 10,
                           n_Q: int = 50, verbose: bool = True) -> dict:
    """
    Validate inverse solver with synthetic data.

    Parameters
    ----------
    true_params : tuple, optional
        True (α, β, γ, δ, Γ). Uses default if None.
    N_a, N_b, N_c : int
        Crystallite dimensions (smaller for speed)
    n_Q : int
        Number of Q points
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Validation results
    """
    if true_params is None:
        true_params = (0.6, 0.4, 0.5, 0.3, 0.03)

    true_alpha, true_beta, true_gamma, true_delta, true_Gamma = true_params

    # Create forward model
    model = PowderDiffractionModel(N_a=N_a, N_b=N_b, N_c=N_c)
    Q_grid = np.linspace(0.5, 4.0, n_Q)

    # Generate synthetic data
    if verbose:
        print("Generating synthetic data...")
        print(f"  True params: α={true_alpha}, β={true_beta}, γ={true_gamma}, "
              f"δ={true_delta}, Γ={true_Gamma}")

    I_obs = model.compute_pattern_fast(
        Q_grid, true_Gamma, true_alpha, true_beta, true_gamma, true_delta)

    # Solve inverse problem
    solver = InverseSolver(model, Q_grid, I_obs)
    result = solver.solve(method='multi_start', n_restarts=3, verbose=verbose)

    # Compare
    errors = {
        'alpha': abs(result.alpha - true_alpha),
        'beta': abs(result.beta - true_beta),
        'gamma': abs(result.gamma - true_gamma),
        'delta': abs(result.delta - true_delta),
        'Gamma': abs(result.Gamma - true_Gamma),
    }

    if verbose:
        print("\nParameter errors:")
        for name, err in errors.items():
            print(f"  {name}: {err:.4f}")

    return {
        'true_params': true_params,
        'estimated_params': result.params,
        'errors': errors,
        'loss': result.loss,
        'success': result.success
    }


def plot_inverse_result(Q_grid: np.ndarray, I_obs: np.ndarray,
                       model: PowderDiffractionModel,
                       result: InverseResult,
                       save_path: str = None):
    """
    Plot observed vs fitted patterns.

    Parameters
    ----------
    Q_grid : np.ndarray
        Q values
    I_obs : np.ndarray
        Observed intensities
    model : PowderDiffractionModel
        Forward model
    result : InverseResult
        Inverse solution
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    # Compute fitted pattern
    I_fit = model.compute_pattern_fast(
        Q_grid, result.Gamma,
        result.alpha, result.beta, result.gamma, result.delta)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [3, 1]})

    # Main plot
    ax1 = axes[0]
    ax1.plot(Q_grid, I_obs, 'k-', linewidth=2, label='Observed', alpha=0.7)
    ax1.plot(Q_grid, I_fit, 'r--', linewidth=1.5, label='Fitted')
    ax1.set_ylabel('I(Q) (arb. units)', fontsize=12)
    ax1.set_title(f'Inverse Problem Result\n'
                 f'α={result.alpha:.3f}, β={result.beta:.3f}, '
                 f'γ={result.gamma:.3f}, δ={result.delta:.3f}, '
                 f'Γ={result.Gamma:.4f}', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2 = axes[1]
    residuals = I_obs - I_fit
    ax2.plot(Q_grid, residuals, 'b-', linewidth=1)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Q (Å⁻¹)', fontsize=12)
    ax2.set_ylabel('Residual', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    return fig, axes


def plot_bo_convergence(result: BayesianInverseResult,
                        save_path: str = None):
    """
    Plot convergence trace from Bayesian optimization.

    Parameters
    ----------
    result : BayesianInverseResult
        Result from BayesianInverseSolver
    save_path : str, optional
        Path to save figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = range(1, len(result.convergence_trace) + 1)
    ax.plot(iterations, result.convergence_trace, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Best Loss (SSE)', fontsize=12)
    ax.set_title(f'Bayesian Optimization Convergence\n'
                 f'Final: α={result.alpha:.3f}, β={result.beta:.3f}, '
                 f'γ={result.gamma:.3f}, δ={result.delta:.3f}', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved convergence plot to {save_path}")
    else:
        plt.show()

    return fig, ax


if __name__ == '__main__':
    import sys

    # Check for Bayesian optimization mode
    use_bo = '--bo' in sys.argv or '-b' in sys.argv

    if use_bo:
        # Validate Bayesian optimization solver
        print("="*60)
        print("Bayesian Optimization Inverse Solver Validation")
        print("="*60)

        if not SKOPT_AVAILABLE:
            print("ERROR: scikit-optimize not available.")
            print("Install with: pip install scikit-optimize")
            sys.exit(1)

        results = validate_bo_solver(
            true_params=(0.7, 0.4, 0.6, 0.5),
            Gamma=0.02,
            n_calls=100,
            N_a=5, N_b=5, N_c=10,
            n_Q=50,
            verbose=True
        )

        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Final loss: {results['loss']:.6e}")
        print(f"Max parameter error: {results['max_error']:.4f}")
        print(f"Total evaluations: {results['n_evaluations']}")
        print(f"Time: {results['elapsed_time']:.1f} s")

    else:
        # Validate original inverse solver
        print("="*60)
        print("Inverse Solver Validation")
        print("="*60)
        print("(Use --bo flag for Bayesian optimization validation)")

        results = validate_inverse_solver(
            true_params=(0.6, 0.4, 0.5, 0.3, 0.03),
            N_a=5, N_b=5, N_c=10,
            n_Q=50,
            verbose=True
        )

        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(f"Success: {results['success']}")
        print(f"Final loss: {results['loss']:.6e}")

        max_error = max(results['errors'].values())
        print(f"Max parameter error: {max_error:.4f}")

        if max_error < 0.1:
            print("PASSED: Parameters recovered within tolerance")
        else:
            print("WARNING: Large parameter errors detected")
