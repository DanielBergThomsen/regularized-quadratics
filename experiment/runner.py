from typing import Union, List, Tuple
import itertools

from tqdm.auto import tqdm
import numpy as np
from scipy.optimize import minimize
import pandas as pd

from experiment.krylov_solvers import krylov_method
from experiment.regularized_quadratic import objective_value, objective_gradient, problem_generator
from experiment.utils import composite_gradient_method, gradient_method, heavy_ball


def run_experiment(
         repetitions: int = 1, 
         min_eig: Union[float, List[float]] = 0, 
         max_eig: Union[float, List[float]] = 1000, 
         regularization: Union[float, List[float]] = 1, 
         power: Union[int, List[int]] = 3, 
         dimensions: Union[int, List[int]] = 100, 
         iterations: Union[int, List[int]] = None,
         solution_dist: Union[float, str, List[float]] = 'worst_case',
         problem_type: Union[str, List[float]] = 'random',
         problem_eigendist: Union[str, List[str]] = None,
         problem_pidist: Union[str, List[str]] = None,
         solvers: Union[str, List[str]] = 'krylov_method',
         save_eigenvalues: bool = False) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """Run a series of experiments with the given parameters.

    This is the backbone that actually collects all the data for the experiments.

    Args:
        repetitions (int, optional): Number of times to repeat experiment.
        (only useful when some randomness involved). Defaults to 1.
        min_eig (Union[float, List[float]], optional): Smallest eigenvalue of A. Defaults to 0.
        max_eig (Union[float, List[float]], optional): Largest eigenvalue of A. Defaults to 1000.
        regularization (Union[float, List[float]], optional): Regularization strength (s). Defaults to 1.
        power (Union[int, List[int]], optional): Power of the regularizer. Defaults to 3.
        dimensions (Union[int, List[int]]): Dimension of problem instance. Defaults to 100.
        iterations (Union[int, List[int]], optional): Maximum number of iterations to run solver. Defaults to None.
        solution_dist (Union[float, str, List[float]], optional): Solution norm. Defaults to 'worst_case'.
        problem_type (Union[str, List[float]], optional): Problem type (cf. problem_generator_factory in regularized_quadratic.py).
        problem_eigendist (Union[str, List[str]], optional): Eigenvalue distribution of the problem. Defaults to None.
        problem_pidist (Union[str, List[str]], optional): Initial point distribution of the problem. Defaults to None.
        solvers (Union[str, List[str]], optional): Solver to use. 
        Possible values are 'krylov_method', 'CG' (conjugate gradient method), 'GD' (gradient descent), 'HB' (heavy-ball method with Polyak step size), 'HBT' ().
        Defaults to 'krylov_method'.
        save_eigenvalues (bool, optional): Whether to save the eigenvalues of the problem instances in a separet dataframe. Defaults to False.

    Raises:
        ValueError: Unknown solver.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]: Dataframe with the results of the experiments.
    """

    # Generate all combinations of parameters, so we can iterate over them as tuples in the next step
    min_eig = [min_eig] if isinstance(min_eig, (int, float)) else min_eig
    max_eig = [max_eig] if isinstance(max_eig, (int, float)) else max_eig
    regularization = [regularization] if isinstance(regularization, (int, float)) else regularization
    power = [power] if isinstance(power, int) or isinstance(power, float) else power
    dimensions = [dimensions] if isinstance(dimensions, int) else dimensions
    iterations = [iterations] if isinstance(iterations, int) or (iterations is None) else iterations
    solution_dist = [solution_dist] if isinstance(solution_dist, (int, float, str)) else solution_dist
    problem_type = [problem_type] if isinstance(problem_type, str) else problem_type
    problem_eigendist = [problem_eigendist] if isinstance(problem_eigendist, str) or (problem_eigendist is None) else problem_eigendist
    problem_pidist = [problem_pidist] if isinstance(problem_pidist, str) or (problem_pidist is None) else problem_pidist
    solvers = [solvers] if isinstance(solvers, str) else solvers
    parameters = itertools.product(
        min_eig, 
        max_eig, 
        regularization, 
        power, 
        dimensions, 
        iterations, 
        solution_dist, 
        problem_type, 
        problem_eigendist, 
        problem_pidist, 
        solvers, 
        list(range(repetitions))
    )

    # Run experiments
    data = []
    total_params = (
        len(min_eig) * 
        len(max_eig) * 
        len(regularization) * 
        len(power) * 
        len(dimensions) * 
        len(iterations) * 
        len(solution_dist) * 
        len(problem_type) *
        len(problem_eigendist) * 
        len(problem_pidist) * 
        len(solvers) * 
        repetitions
    )
    eigenvals = {}
    for (
        mu, L, s, p, d, N, x_dist, problem_type, eigendist, pidist, solver, seed
    ) in tqdm(parameters, total=total_params, desc='Running experiments', position=0, leave=False):

        # Make sure there is actually something random to use this run for, otherwise skip
        if (
            seed != 0 and 
            (eigendist in ['heuristic', 'adversarial', 'uniform']) and 
            (pidist in ['heuristic', 'adversarial', 'uniform']) and 
            (problem_type == 'adversarial')
        ):
            continue

        # Set seed
        rng = np.random.default_rng(seed)

        # Compute x_dist if indicated
        N = int((d - 1) / 2) if N is None else N
        if x_dist == 'worst_case':
            x_dist_value = (L / (3 * s * N**2))**(1 / (p - 2))
        else:
            x_dist_value = x_dist

        # Generate problem instance
        problem_constructor = problem_generator(problem_type, eigendist, pidist, d, N)
        problem_instance, solution = problem_constructor(mu, L, s, p, d, x_dist_value, rng)

        # Set initial point
        if problem_type == 'one_step_construction':
            smallest_eigenvec_A = np.linalg.eigh(problem_instance[0])[1][:, 0]
            initial_point = x_dist_value * smallest_eigenvec_A
        else:
            initial_point = np.zeros(d)

        # Solve problem instance and collect residuals
        optimal_value = objective_value(problem_instance, solution)
        if solver == 'krylov_method':
            fs, norm_dist = krylov_method(problem_instance, solution, initial_point, N)
            fres = fs - optimal_value
        elif solver == 'CG':
            xs = minimize(lambda x: objective_value(problem_instance, x), 
                            initial_point, 
                            method='CG', 
                            jac=lambda x: objective_gradient(problem_instance, x),
                            options = {'gtol': 1e-30, 'disp': True, 'maxiter': N, 'return_all': True}).allvecs

            norm_dist = np.array([np.linalg.norm(x - solution) for x in xs])
            fres = np.array([objective_value(problem_instance, x) - optimal_value for x in xs])
        elif solver == 'GD':
            fs, norm_dist = gradient_method(problem_instance, initial_point, solution, N, 
                                             adaptive=problem_type != 'one_step_construction')  # Our proof uses a simpler step size here
            fres = fs - optimal_value
        elif solver == 'CGM':
            fs, norm_dist = composite_gradient_method(problem_instance, initial_point, solution, N)
            fres = fs - optimal_value
        elif solver == 'HB':
            fs, norm_dist = heavy_ball(problem_instance, initial_point, solution, N)
            fres = fs - optimal_value
        else:
            raise ValueError(f'Unknown solver: {solver}')

        # Store residuals
        for i, (fres, x_norm_dist) in enumerate(zip(fres, norm_dist)):
            data.append({'min_eig': mu, 
                         'max_eig': L, 
                         'regularization_strength': s, 
                         'power': p, 
                         'dimension': d, 
                         'max_iterations': N,
                         'solution_norm': x_dist_value, 
                         'functional_residual': fres, 
                         'norm_dist': x_norm_dist, 
                         'iteration': i,
                         'problem_type': problem_type,
                         'eigendist': eigendist,
                         'pidist': pidist,
                         'seed': seed,
                         'method': solver})

        # Add eigenvalues if indicated
        if save_eigenvalues:
            eigenvals[(mu, L, s, p, d, x_dist, eigendist, pidist)] = np.linalg.eigvals(problem_instance[0])

    # Make into dataframe and return
    df = pd.DataFrame(data)
    return (df, eigenvals) if save_eigenvalues else df