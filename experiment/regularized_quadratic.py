import numpy as np
import scipy
from scipy.stats import ortho_group
import cvxpy as cvx

from typing import Callable, Iterable, Tuple, Union


def problem_generator(problem_type: str, eigendist: str, pidist: str, dimensions: int, N: int = None) -> Callable:
    """Helps create a problem generator function that can be used to generate problems of a certain type.

    Args:
        problem_type (str): 'random', 'adversarial' or 'one_step_construction'.
        eigendist (str): Left at none for default behavior on random and adversarial problems.
        'uniform' for uniform distribution between 0 and 1.
        'beta_a_b' for beta distribution with parameters a and b.
        pidist (str): Left at none for default behavior on random and adversarial problems.
        'uniform' for uniform distribution between 0 and 1.
        'dirichlet_alpha' for dirichlet distribution with parameter vector full of single value alpha.
        dimensions (int): Number of dimensions.
        N (int, optional): Number of eigenvalues to be used in the construction. Defaults to None.

    Raises:
        ValueError: If an unknown eigendist is given.

    Returns:
        function: A function that generates a problem instance given the parameters.
    """

    # Default behavior
    if eigendist is None or (eigendist == 'adversarial'):
        eigendist = None
    elif eigendist == 'uniform':
        eigendist = scipy.stats.uniform(loc=0, scale=1)
    elif eigendist.startswith('beta'):
        tmp = eigendist.split('_')
        a, b = float(tmp[1]), float(tmp[2])
        eigendist = scipy.stats.beta(a, b)
    else:
        raise ValueError(f"Unknown eigendist: {eigendist}")

    if pidist is None:
        pass
    elif pidist == 'uniform':
        pidist = np.full(dimensions, 1 / dimensions)
    elif pidist.startswith('dirichlet_'):
        tmp = pidist.split('_')
        alpha = float(tmp[1])
        pidist = scipy.stats.dirichlet(alpha * np.ones(dimensions))

    # Just pass these arguments to corresponding problem constructor
    if problem_type == 'random':
        return lambda mu, L, s, p, d, xopt_norm, rng: generate_RQP(mu, L, s, p, d, xopt_norm, rng, eigendist)
    elif problem_type == 'adversarial':
        return lambda mu, L, s, p, d, xopt_norm, rng: generate_construction(mu, L, s, p, d, xopt_norm, rng, pidist, eigendist, N)
    elif problem_type == 'one_step_construction':
        return lambda mu, L, s, p, d, xopt_norm, rng: generate_one_step_construction(mu, L, s, p, d, xopt_norm, rng, eigendist)

def chebyshev_extrema(L_star: float, mu_star: float, dimensions: int) -> np.ndarray:
    """Generates Chebyshev extrema for a given interval.

    Args:
        L_star (float): Value of L + sr^{p-2} in theory.
        mu_star (float): Value of mu + sr^{p-2} in theory.
        dimensions (int): Number of dimensions.

    Returns:
        np.ndarray: Chebyshev extrema.
    """
    return np.array([1 / 2 * (L_star + mu_star - np.cos(k * np.pi / (dimensions-1))*(L_star - mu_star)) 
                     for k in range(dimensions)])

def projection_simplex_sort(v: Iterable[float], z: float = 1) -> np.ndarray:
    """Euclidean projection of a vector onto the simplex using sorting.

    For details check out 'Large-scale Multiclass Support Vector Machine Training via Euclidean Projection 
    onto the Simplex' (https://mblondel.org/publications/mblondel-icpr2014.pdf)

    Args:
        v (Iterable[float]): Vector to be projected.
        z (float, optional): Sum of the elements of the vector. Defaults to 1.

    Returns:
        np.ndarray: Projected vector.
    """
    v = np.array(v)
    d = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(d) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return np.array(w)

def _c_opt(pi: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Computes the optimal c for a given pi and V.

    This is a helper function _cost and _cost_grad.

    Args:
        pi (np.ndarray): Vector pi of probabilities.
        V (np.ndarray): Vandermonde matrix.

    Returns:
        np.ndarray: Optimal c.
    """
    pi_m = np.diag(pi)
    A_inv = np.linalg.pinv(V.T @ pi_m @ V)
    return A_inv @ V.T @ pi

def _cost(pi: np.ndarray, V: np.ndarray) -> float:
    """Computes the cost function for a given pi and V.

    This can be used to try to directly solve the optimization problem associated with pi in our construction.
    However, we recommend directly using the CVX implementation to do it instead. This is just for reference.

    Args:
        pi (np.ndarray): Vector pi of probabilities.
        V (np.ndarray): Vandermonde matrix.

    Returns:
        float: Cost value.
    """
    pi_m = np.diag(pi)
    return -np.linalg.norm(pi_m @ (np.ones_like(pi) - V @ _c_opt(pi, V)), ord=2)**2

def _cost_grad(pi: np.ndarray, V: np.ndarray, eigenvals: np.ndarray) -> np.ndarray:
    """Computes the gradient of the cost function for a given pi and V.

    This can be used to try to directly solve the optimization problem associated with pi in our construction.
    However, we recommend directly using the CVX implementation to do it instead. This is just for reference.

    Args:
        pi (np.ndarray): Vector pi of probabilities.
        V (np.ndarray): Vandermonde matrix.
        eigenvals (np.ndarray): Eigenvalues of the matrix.

    Returns:
        np.ndarray: Gradient of the cost function.
    """
    return -2 * (np.ones_like(pi) - eigenvals * (V @ _c_opt(pi, V))) ** 2

def find_pi_cvx(eigenvals: np.ndarray, spn_replacement: str = 'QR', eps: float = 1e-3, verbose: bool = False) -> np.ndarray:
    """Finds the pi vector using CVX.

    The original problem of finding pi can be formulated as a convex optimization problem.
    We use CVX to solve this problem in this function.

    Args:
        eigenvals (np.ndarray): Eigenvalues of the matrix.
        spn_replacement (str, optional): Pre-transforms the matrix V into something with the same span. 
        Can be set to 'QR' or 'none'. Defaults to 'QR'.
        eps (float, optional): Tolerance for the solver. Defaults to 1e-3.
        verbose (bool, optional): Whether to print the solver output. Defaults to False.

    Returns:
        np.ndarray: Pi vector found by CVX.
    """
    d = len(eigenvals) - 1
    V = np.vander(eigenvals[:d+1], increasing=True)[:, 1:d+1]

    # Helps with conditioning to use QR decomposition
    if spn_replacement == 'QR':
        V, _ = np.linalg.qr(V)

    # Reformulate the problem. See the appendix of the paper for details.
    pi = cvx.Variable(d+1)
    X = cvx.Variable((d+1, d+1), symmetric=True)
    constraints = [pi >= 0, cvx.sum(pi) == 1, X >> 0,
                   X[1:, 0] == V.T @ pi,
                   X[1:, 1:] == V.T @ cvx.diag(pi) @ V]
    objective = cvx.Minimize(X[0, 0])
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver='CLARABEL',   # CLARABEL works well for this problem.
               tol_gap_abs = eps, 
               tol_gap_rel = eps, 
               tol_feas = eps, 
               equilibrate_enable = True,
               iterative_refinement_enable = True,
               min_terminate_step_length = eps,
               verbose=verbose)
    pi = pi.value
    pi[pi < 0] = 0
    return pi
    
def last_coef(n, c1=2.0030004460202253, c2=0.3785972015639097, c3=-9.519199185333344, c4=0.001235242856667443) -> float:
    """Computes the last coefficient of the pi vector.

    Coefficients fit by generating solutions using CVX and then fitting the coefficients (cf. 'estimating_pi.ipynb').

    Args:
        n (int): Number of dimensions.
        c1 (float, optional): Defaults to 2.0030004460202253.
        c2 (float, optional): Defaults to 0.3785972015639097.
        c3 (float, optional): Defaults to -9.519199185333344.
        c4 (float, optional): Defaults to 0.001235242856667443.

    Returns:
        float: Last coefficient of the pi vector.
    """
    return c1**(-c2*(n-c3))+c4

def first_coef(n, c1=2.0824014739014203, c2=0.9961791558193664, c3=0.41050294562291606, c4=0.42600453090655127):
    """Computes the first coefficient of the pi vector.

    Coefficients fit by generating solutions using CVX and then fitting the coefficients (cf. 'estimating_pi.ipynb').

    Args:
        n (int): Number of dimensions.
        c1 (float, optional): Defaults to 2.0824014739014203.
        c2 (float, optional): Defaults to 0.9961791558193664.
        c3 (float, optional): Defaults to 0.41050294562291606.
        c4 (float, optional): Defaults to 0.42600453090655127.
    
    Returns:
        float: First coefficient of the pi vector.
    """
    return c1**(-c2*(n-c3))+c4

def middle_coefs(n, c1 = 0.060765593865534206, c2=1.8430375329834494, c3=1.6991040151256829):
    """Computes the middle coefficients of the pi vector.

    Coefficients fit by generating solutions using CVX and then fitting the coefficients (cf. 'estimating_pi.ipynb').

    Args:
        n (int): Number of dimensions.
        c1 (float, optional): Defaults to 0.060765593865534206.
        c2 (float, optional): Defaults to 1.8430375329834494.
        c3 (float, optional): Defaults to 1.6991040151256829.

    Returns:
        np.ndarray: Middle coefficients of the pi vector.
    """
    xs = c2 * (np.arange(1, n) + c1)
    return xs**(-c3)

def pi_hypothesis(n: int) -> np.ndarray:
    """Computes the pi vector the models fit in 'estimating_pi.ipynb'.

    Args:
        n (int): Number of dimensions.

    Returns:
        np.ndarray: Pi vector.
    """
    pi_0 = first_coef(n)
    pi_last = last_coef(n)
    pi_middle = middle_coefs(n-1)
    pi_middle = (1 - pi_0 - pi_last) * pi_middle / np.sum(pi_middle)  # Normalize to sum to 1
    pi = np.concatenate(([pi_0], pi_middle, [pi_last]))
    return pi


def generate_construction(min_eig: float, 
                          max_eig: float, 
                          regularization: float, 
                          power: int, 
                          dimensions: int, 
                          solution_norm: float, 
                          rng: np.random.Generator,
                          pidist: scipy.stats.rv_continuous = None, 
                          eigendist: scipy.stats.rv_continuous = None,
                          N: int = None) -> Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]:
    """Generates construction in accordance with the theory presented in the paper.

    Note: rotations need to be constructed adversarially for A on the fly during the 
    optimization process to formally adhere to the construction.
    In our experiments this is not a problem because we primarily use Krylov subspace methods.

    Args:
        min_eig (float): Minimum eigenvalue.
        max_eig (float): Maximum eigenvalue.
        regularization (float): Regularization parameter.
        power (int): Power of the regularizer.
        dimensions (int): Number of dimensions.
        solution_norm (float): Norm of the solution.
        rng (np.random.Generator): Random number generator.
        pidist (scipy.stats.rv_continuous, optional): Distribution of pi. Defaults to None.
        Example pi_dist: pi_dist = np.random.dirichlet(np.ones(dimensions))
        eigendist (scipy.stats.rv_continuous, optional): Distribution of eigenvalues. 
        Example eigendist: eigendist = np.random.beta(0.49, 0.49, size=dimensions)
        Defaults to None.
        N (int, optional): Number of eigenvalues to be used in the construction. Defaults to None.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]: Problem instance (A, b, s, p) and optimal solution.
    """

    sr = regularization * solution_norm**(power - 2)
    mu_star = min_eig + sr
    L_star = max_eig + sr

    # Generate eigenvalues placed at the Chebyshev extrema
    if eigendist is None:
        eigenvalues = chebyshev_extrema(L_star, mu_star, 2*N + 1) - sr

    # Generate eigenvalues from a given distribution with support [a, b]
    else:
        eigenvalues = eigendist.rvs(random_state=rng, size=dimensions).reshape(-1)
        eigenvalues = eigenvalues[:2*N+1]  # Only keep 2*N+1 eigenvalues
        lower, upper = eigendist.a, eigendist.b
        eigenvalues = eigenvalues * (L_star - mu_star) / (upper - lower) - lower / (upper - lower) + mu_star - sr

    eigenvalues = np.concatenate((eigenvalues, np.zeros(dimensions - 2*N - 1)))
    A = np.diag(eigenvalues)

    if isinstance(pidist, np.ndarray):
        pi_sqrt = np.sqrt(pidist)

    # Sample pi from a given scipy distribution
    elif isinstance(pidist, scipy.stats.rv_continuous):
        pi_sqrt = np.sqrt(pidist.rvs(random_state=rng).reshape(-1))

    # Generate pi according to our heuristic
    elif pidist == 'heuristic':
        eigenstars = eigenvalues[:2*N+1].copy()
        eigenstars[:2*N+1] += sr
        pi = np.zeros(dimensions)
        pi[:2*N+1] = pi_hypothesis(2*N+1)
        pi_sqrt = np.sqrt(pi)

    # Computes pi numerically using CVX
    elif pidist == 'adversarial':
        eigenstars = eigenvalues[:2*N+1].copy()
        eigenstars[:2*N+1] += sr
        spn_replacement = 'QR'  # Helps with conditioning
        pi = np.zeros(dimensions)
        pi[:2*N+1] = find_pi_cvx(eigenstars, spn_replacement=spn_replacement, eps=1e-8, verbose=False)
        pi_sqrt = np.sqrt(pi)
    
    else:
        raise ValueError(f"Unknown pidist: {pidist}")

    # Set b according to construction
    b = solution_norm * (A + regularization * solution_norm**(power - 2) * np.eye(dimensions)) @ pi_sqrt
    x_opt = solution_norm * pi_sqrt
    return (A, b, regularization, power), x_opt


def generate_RQP(min_eig: float, 
                 max_eig: float, 
                 regularization: float, 
                 power: int, 
                 dimensions: int, 
                 solution_dist: float, 
                 rng: np.random.Generator,
                 eigen_dist: Union[scipy.stats.rv_continuous, str] = 'even') -> Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]:
    """Generates a random quadratic problem.

    Construction is generated by sampling eigenvalues from a given distribution and then setting
        A = U^T D U 
    where U is a random orthogonal matrix (sampled uniformly using Haar distribution) and D is a 
    diagonal matrix with the sampled eigenvalues. The solution is then generated by sampling a
    random vector from a unit ball and normalizing it to have the desired norm solution_dist.

    Args:
        min_eig (float): Minimum eigenvalue.
        max_eig (float): Maximum eigenvalue.
        regularization (float): Regularization parameter.
        power (int): Power of the regularizer.
        dimensions (int): Number of dimensions.
        solution_dist (float): Solution norm.
        rng (np.random.Generator): Random number generator.
        eigen_dist (scipy.stats.rv_continuous, optional): Distribution of eigenvalues. Defaults to 'even'.
    
    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]: Problem instance (A, b, s, p) and optimal solution.
    """

    # Partition eigenvalues evenly spaced between min and max if no eigen_dist is given
    if eigen_dist is None or (eigen_dist == 'even'):
        eigenvalues = np.linspace(min_eig, max_eig, dimensions)
    else:
        eigenvalues = eigen_dist.rvs(size=dimensions, random_state=rng)

    # Make sure the smallest and largest eigenvalues are exactly min_eig and max_eig
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = np.concatenate((np.array([min_eig]), eigenvalues[1:-1], np.array([max_eig])))

    # Generate A
    A = np.diag(eigenvalues)
    u = ortho_group.rvs(dimensions, random_state=rng)
    A = u.T @ A @ u  # Rotation will be uniformly distributed

    # Generate x*
    coords = rng.normal(size=dimensions)
    distance_from_origin = np.linalg.norm(coords, ord=2)
    x_opt = solution_dist * coords / distance_from_origin

    # b is now given by a closed form expression
    b = (A + regularization * solution_dist**(power - 2) * np.identity(dimensions)) @ x_opt
    return (A, b, regularization, power), x_opt

def generate_one_step_construction(min_eig: float, 
                 max_eig: float, 
                 regularization: float, 
                 power: int, 
                 dimensions: int, 
                 solution_dist: float, 
                 rng: np.random.Generator,
                 eigen_dist: Union[scipy.stats.rv_continuous, str] = 'even') -> Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]:
    """Generates a quadratic problem according to the construction used for our lower bound on one-step methods.

    Args:
        min_eig (float): Minimum eigenvalue.
        max_eig (float): Maximum eigenvalue.
        regularization (float): Regularization parameter.
        power (int): Power of the regularizer.
        dimensions (int): Number of dimensions.
        solution_dist (float): Solution norm.
        rng (np.random.Generator): Random number generator.
        eigen_dist (Union[scipy.stats.rv_continuous, str], optional): Distribution of eigenvalues. Defaults to 'even'.

    Returns:
        Tuple[Tuple[np.ndarray, np.ndarray, float, float], np.ndarray]: Problem instance (A, b, s, p) and optimal solution.
    """
    # Partition eigenvalues evenly spaced between min and max if no eigen_dist is given
    if eigen_dist is None or (eigen_dist == 'even'):
        eigenvalues = np.linspace(min_eig, max_eig, dimensions)
    else:
        eigenvalues = eigen_dist.rvs(size=dimensions, random_state=rng)

    # Make sure the smallest and largest eigenvalues are exactly min_eig and max_eig
    eigenvalues = np.sort(eigenvalues)
    eigenvalues = np.concatenate((np.array([min_eig]), eigenvalues[1:-1], np.array([max_eig])))

    # Generate A
    A = np.diag(eigenvalues)
    u = ortho_group.rvs(dimensions, random_state=rng)
    A = u.T @ A @ u  # Rotation will be uniformly distributed

    # Generate x*
    x_opt = np.zeros(dimensions)
    b = np.zeros(dimensions)

    return (A, b, regularization, power), x_opt

def objective_value(problem_instance: Tuple[np.ndarray, np.ndarray, float, float], x: np.ndarray) -> float:
    """Computes the objective value of a given problem instance at a given point.

    f(x) = 1/2 x^T A x - b^T x + regularization / power * ||x||_2^p

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, float]): Problem instance (A, b, s, p).
        x (np.ndarray): Point at which to evaluate the objective.

    Returns:
        float: Objective value at x.
    """
    A, b, regularization, power = problem_instance
    return 1 / 2 * x.T @ A @ x - b.T @ x + regularization / power * np.linalg.norm(x, ord=2)**power

def objective_gradient(problem_instance: Tuple[np.ndarray, np.ndarray, float, float], x: np.ndarray) -> np.ndarray:
    """Computes the gradient of the objective function at a given point.

    \nabla f(x) = A x - b + regularization * ||x||_2^(p-2) * x

    Args:
        problem_instance (Tuple[np.ndarray, np.ndarray, float, float]): Problem instance (A, b, s, p).
        x (np.ndarray): Point at which to evaluate the gradient.

    Returns:
        np.ndarray: Gradient of the objective at x.
    """
    A, b, regularization, power = problem_instance
    return A @ x - b + regularization * np.linalg.norm(x, ord=2)**(power - 2) * x