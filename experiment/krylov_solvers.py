import numpy as np
from tqdm.auto import tqdm
import scipy.optimize

from experiment.regularized_quadratic import objective_value
from typing import Callable, Tuple


def generate_krylov_objective(problem_instance: Tuple[np.ndarray, np.ndarray, float, int], 
                              krylov_matrix: np.ndarray, 
                              order: int) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    """Generates the objective function and its gradient for the Krylov subspace method subproblem.

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, float]): Problem instance (A, b, s, p).
        krylov_matrix (np.ndarray): Krylov matrix.
        order (int): Order of the Krylov subspace.

    Returns:
        Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]: Objective function and its gradient.
    """
    A, b, regularization, power = problem_instance
    def objective(coefs):
        x = krylov_matrix[:, :order] @ coefs
        return objective_value(problem_instance, x)

    def grad(coefs):
        K = krylov_matrix[:, :order]
        x = K @ coefs
        return (K.T @ (A @ x - b) + regularization * np.linalg.norm(x, ord=2)**(power - 2) * K.T @ x)
    return objective, grad


def lanczos_method(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lanczos method for generating a matrix with equivalent basis to a Krylov matrix.

    This is used to generate a Krylov matrix that has good conditioning properties.

    Args:
        A (np.ndarray): Matrix A.
        b (np.ndarray): Vector b.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Basis vectors qs, diagonal elements and off-diagonal 
        elements of the Krylov matrix.
    """
    d = len(b)
    qs = [np.zeros(d), b / np.linalg.norm(b, ord=2)]
    diagonals = []
    off_diagonals = [0]

    for i in range(1, d + 1):
        alpha = qs[i].T @ A @ qs[i]
        diagonals.append(alpha)
        if i == len(b):
            break
        temp = A @ qs[i] - alpha * qs[i] - off_diagonals[i - 1] * qs[i - 1]
        beta = np.linalg.norm(temp, ord=2)
        off_diagonals.append(beta)
        qs.append(temp / beta)

    return qs[1:], diagonals, off_diagonals[1:]


def krylov_method(problem_instance: Tuple[np.ndarray, np.ndarray, float, int],
                  solution: np.ndarray,
                  initial_point: np.ndarray,
                  N: int,
                  basis: str = "lanczos",
                  linear_solver: bool = True,
                  progress_bar: bool = True) -> np.ndarray:
    """Krylov subspace method for solving the regularized quadratic problem.

    Each iteration of the Krylov subspace method solves a subproblem of the form:
        min f(x) = 1/2 x^T A x - b^T x + s/3 ||x||^3,
    but with x in the Krylov subspace spanned by the columns of the i-th order Krylov matrix.

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, int]): Problem instance (A, b, s, p).
        solution (np.ndarray): Optimal solution to the problem instance.
        initial_point (np.ndarray): Initial point.
        xopt (np.ndarray): Optimal solution.
        N (int): Number of iterations.
        basis (str, optional): Basis to use for generating the Krylov matrix. Defaults to "lanczos".
        linear_solver (bool, optional): Whether to use the linear solver. Defaults to True.
        adversarial (bool, optional): Whether to use adversarial rotations. Defaults to False.
        progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

    Raises:
        Exception: Rotation was not orthogonal.
        Exception: Rotation was not b-invariant.

    Returns:
        np.ndarray: Objective values at each iteration.
    """
    A, b, regularization, power = problem_instance
    fs = np.zeros(N+1)
    fs[0] = objective_value(problem_instance, initial_point)
    norm_dist = np.zeros(N+1)
    norm_dist[0] = np.linalg.norm(initial_point - solution, ord=2)

    # Choose basis depending on parameter
    # NOTE: natural basis has terrible conditioning, so we use Lanczos basis by default
    if basis == "natural":
        Q = np.array([(np.linalg.matrix_power(A, i) @ b) / np.linalg.norm((np.linalg.matrix_power(A, i) @ b), ord=2) for i in range(N)]).T
    elif basis == "lanczos":
        qs, _, _ = lanczos_method(A, b)
        Q = np.array(qs).T
        
    progress_bar = tqdm(range(1, N+1), disable=not progress_bar, desc='Krylov subspace solver', position=2, leave=False)
    x = np.array([])
    initial_r = None
    for i in progress_bar:
        qf, qgr = generate_krylov_objective(problem_instance, Q, order = 2*i)
        Q_2i = Q[:, :2*i]

        # Optimize objectives directly (using BFGS) or use Newton routine
        if linear_solver:
            x, _, _, initial_r = solver_step(Q_2i.T @ b, Q_2i.T @ A @ Q_2i, regularization, eps = 1e-10, p=power, initial_r=initial_r)
        else:
            initial_p = np.zeros(2*i)
            if len(x) > 0:
                initial_p[:len(x)] = x
            result = scipy.optimize.minimize(qf, initial_p, method='BFGS', 
                            jac = qgr, 
                            options = {'gtol': 1e-20, 'disp': False})
            x = result.x

        iteration = Q_2i @ x
        fres = objective_value(problem_instance, iteration)
        fs[i] = objective_value(problem_instance, iteration)
        norm_dist[i] = np.linalg.norm(iteration - solution, ord=2)

        # Stop if the objective value is increasing (can happen with bad conditioning)
        if (i > 4) and (fres > fs[i-1]):
            fs[i:] = np.nan
            norm_dist[i:] = np.nan
            break
    return fs, norm_dist


def solver_step(b: np.ndarray, 
                A: np.ndarray, 
                s: float, 
                eps: float = 1e-8, 
                p: float = 3, 
                initial_r: float = None) -> Tuple[np.ndarray, float, str, float]:
    """Uses a Newton's method to solve a regularized quadratic.

    Minimizes the function
       f(x) = 1/2 x^T A x - b^T x + s/3 ||x||^3
    by finding the root of the equation:
       h(r) = r - ||(A + sr**(p-2))^{-1} b|| = 0.

    Args:
        b (np.ndarray): Vector b.
        A (np.ndarray): Matrix A.
        s (float): Regularization parameter.
        eps (float, optional): Tolerance. Defaults to 1e-8.
        p (float, optional): Power of the regularization term. Defaults to 3.
        initial_r (float, optional): Initial value for r. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, float, str, float]: Optimal solution, optimal value, status
    """
    d = b.shape[0]

    def f(x, x_norm):
        return 1 / 2 * x.T @ A @ x - b.T @ x + s / p * (x_norm ** p)
    
    def h(r, der=False):
        ArB_cho_factor = scipy.linalg.cho_factor(A + s * r**(p-2) * np.eye(d), lower=False)
        x = scipy.linalg.cho_solve(ArB_cho_factor, b)
        x_norm = (x @ x) ** 0.5
        h_r = r - x_norm
        if der:
            h_r_prime = 1 + s * (p - 2) * x_norm**(p - 4) * \
                        scipy.linalg.cho_solve(ArB_cho_factor, x) @ x
        else:
            h_r_prime = None
        return h_r, x_norm, x, h_r_prime

    max_r = 1.0 if initial_r is None else initial_r
    max_iters = 500

    # Find max_r such that h(max_r) is nonnegative
    for i in range(max_iters):
        h_r, x_norm, x, _ = h(max_r)
        if h_r < -eps:
            max_r *= 2
        elif -eps <= h_r <= eps:
            return x, f(x, x_norm), "success", max_r
        else:
            break
    
    # Univariate Newton's
    r = max_r
    for i in range(max_iters):
        h_r, x_norm, x, h_r_prime = h(r, der=True)
        if -eps <= h_r <= eps:
            return x, f(x, x_norm), "success", r
        r -= h_r / h_r_prime
           
    return np.zeros(d), 0.0, "iterations_exceeded", None