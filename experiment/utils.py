from typing import Tuple
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize

from experiment.regularized_quadratic import objective_value, objective_gradient


def heavy_ball(problem_instance: Tuple[np.ndarray, np.ndarray, float, int],
               initial_point: np.ndarray,
               xopt: np.ndarray, 
               N: int) -> np.ndarray:
    """Heavy-ball method for solving regularized quadratic optimization problems.

    The implementation uses the Polyak step size
        \eta = \frac{2(f_i - f^*)}{\| \nabla f_i \|_2^2}

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, int]): Problem instance (A, b, s, p).
        initial_point (np.ndarray): Initial point.
        xopt (np.ndarray): Optimal point.
        N (int): Number of iterations.

    Returns:
        np.ndarray: Array with the objective values at each iteration.
    """
    fs = np.zeros(N+1)
    fs[0] = objective_value(problem_instance, initial_point)
    norm_dist = np.zeros(N+1)
    norm_dist[0] = np.linalg.norm(xopt - initial_point, ord=2)

    fopt = objective_value(problem_instance, xopt)
    x = initial_point
    prev_x = initial_point
    f_i = objective_value(problem_instance, x)
    grad_x = objective_gradient(problem_instance, x)
    h_i = 2*(f_i - fopt) / np.linalg.norm(grad_x, ord=2)**2
    m_i = 0
    for i in range(1, N+1):

        # Heavy-ball update
        momentum = m_i * (x - prev_x)
        prev_x = x.copy()
        x = x - (1 + m_i) * h_i *  grad_x + momentum

        # Update parameters
        prev_f = f_i.copy()
        prev_grad = grad_x.copy()
        f_i = objective_value(problem_instance, x)
        fs[i] = f_i
        norm_dist[i] = np.linalg.norm(xopt - x, ord=2)
        grad_x = objective_gradient(problem_instance, x)
        h_i = 2 * (f_i - fopt) / np.linalg.norm(grad_x, ord=2)**2
        m_i = -(f_i - fopt) * grad_x.dot(prev_grad) / ((prev_f - fopt) * np.linalg.norm(grad_x, ord=2)**2 + (f_i - fopt) * grad_x.dot(prev_grad))

    return fs, norm_dist



def gradient_method(problem_instance: Tuple[np.ndarray, np.ndarray, float, int],
                     initial_point: np.ndarray, 
                     xopt: np.ndarray, 
                     N: int, 
                     adaptive=True,
                     constant_step_size=None,
                     progress_bar: bool = True) -> np.ndarray:
    """Gradient descent method for solving regularized quadratic optimization problems.

    Step size is the one we argue for in the upper bound of the paper:
        \eta = \frac{1}{L + 4s \| x_\star \|_2}  TODO: update

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, int]): Problem instance (A, b, s, p).
        initial_point (np.ndarray): Initial point.
        xopt (np.ndarray): Optimal point.
        N (int): Number of iterations.
        adaptive (bool, optional): Whether to use step size from our convergence guarantee, 
        constant_step_size (optional): If not None, use this step size instead of the one from the convergence guarantee.
        or just the Lipschitz constant. Defaults to True.
        progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

    Returns:
        np.ndarray: Array with the objective values at each iteration.
    """
    A, b, regularization, power = problem_instance
    r = np.linalg.norm(initial_point - xopt, ord=2)
    fs = np.zeros(N+1)
    fs[0] = objective_value(problem_instance, initial_point)
    norm_dist = np.zeros(N+1)
    norm_dist[0] = np.linalg.norm(xopt - initial_point, ord=2)

    L = np.linalg.eigvalsh(A)[-1]
    x = initial_point
    progress_bar = tqdm(range(1, N+1), disable=not progress_bar)
    for i in progress_bar:
        grad = objective_gradient(problem_instance, x)
        constant_step_size = 1 / (L + regularization * (power - 1) * 2**(power - 2) * r**(power - 2)) \
            if adaptive else 1 / (L + regularization*r**(power - 2))
        alt_step_size = (power / (regularization * 2**(power - 2) * np.linalg.norm(grad, ord = 2)**(power - 2)))**(1 / (power - 1))
        if alt_step_size < constant_step_size:
            print(f"USING SMALL STEP SIZE ON ITERATION: {i}")
        step_size = min(constant_step_size, alt_step_size) if adaptive else constant_step_size
        x = x - step_size * grad
        fs[i] = objective_value(problem_instance, x)
        norm_dist[i] = np.linalg.norm(xopt - x, ord=2)

        # Stop if the objective value is increasing
        if (i > 4) and (fs[i] > fs[i-1]):
            fs[i:] = np.nan
            norm_dist[i:] = np.nan
            break
    return fs, norm_dist


def composite_gradient_method(problem_instance: Tuple[np.ndarray, np.ndarray, float, int],
                               initial_point: np.ndarray, 
                               xopt: np.ndarray, 
                               N: int, 
                               progress_bar: bool = True) -> np.ndarray:
    """Composite gradient method for solving regularized quadratic optimization problems.

        The step size is set to 1 / L by default in accordance with upper bound from (Nesterov 2022).

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, int]): Problem instance (A, b, s, p).
        initial_point (np.ndarray): Initial point.
        xopt (np.ndarray): Optimal point.
        N (int): Number of iterations.
        progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

    Returns:
        np.ndarray: Array with the objective values at each iteration.
    """
    A, b, regularization, power = problem_instance
    fs = np.zeros(N+1)
    fs[0] = objective_value(problem_instance, initial_point)
    norm_dist = np.zeros(N+1)
    norm_dist[0] = np.linalg.norm(xopt - initial_point, ord=2)

    L = np.linalg.eigvalsh(A)[-1]
    x = initial_point
    progress_bar = tqdm(range(1, N+1), disable=not progress_bar)
    for i in progress_bar:

        # Minimize function using scipy
        def quadratic_approx(x_):
            return (A @ x - b).T @ (x_ - x) + 0.5 * L * np.linalg.norm(x_ - x, ord=2) ** 2 + regularization / power * np.linalg.norm(x_, ord=2) ** power
        res = minimize(quadratic_approx,
                       x, 
                       #method='LBFGS', 
                       options = {'gtol': 1e-30})
        x = res.x
        fs[i] = objective_value(problem_instance, x)
        norm_dist[i] = np.linalg.norm(xopt - x, ord=2)

        # Stop if the objective value is increasing
        if (i > 4) and (fs[i] > fs[i-1]):
            fs[i:] = np.nan
            norm_dist[i:] = np.nan
            break
    return fs, norm_dist