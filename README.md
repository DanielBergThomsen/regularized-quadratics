# Complexity of Minimizing Regularized Convex Quadratic Functions
This repo contains the implementation of the experiments in the paper "Complexity of Minimizing Regularized Convex Quadratic Functions" by [Daniel Berg Thomsen](mailto:danielbergthomsen@gmail.com) and [Nikita Doikov](https://www.doikov.com). For the moment, the paper is available on arXiv: https://arxiv.org/abs/2404.17543


## How to reproduce the figures from the paper
### Dependencies
The following will download and install the necessary dependencies:
```bash
pip install -r requirements.txt
```

### Running paper experiments
To generate a given figure from the paper, simply execute the given file directly in python (or run each cell it it's a jupyter notebook) without any arguments: 
- Figure 1: `experiment_upper_bound.py`
- Figure 2: `experiment_one_step_methods.py`
- Figure 3 (a and b): `experiment_functional_residual.py`
- Figure 4: `experiment_lower_bound.py`
- Figure 5: `experiment_lower_bound_uniform_pi.py`
- Figure 6 (a and b): `experiment_estimated_eigenvalues.py`
- Figure 7: `experiment_grid.py`
- Figure 8 (a, b, c) and 9: `estimating_pi.ipynb`

After finishing execution, the figures will be saved in the `figures` directory.

### Running own experiments
All the experiments have adjustable parameters that can be set either in the script or by passing them as arguments. The following is an example of how to run an experiment with different parameters:
```bash
python experiment_lower_bound.py \
            --power 6 \
            --figure_path "figures/lower_bound_p=6.pdf"
```
Note that the plotting functions have been written with a specific set of parameters in mind and may have to be adjusted.

## References
If you find this repo or the paper itself to be relevant for your work, please consider citing our paper. For now, you can use the following BibTeX entry:
```
@misc{bergthomsen2024complexity,
      title={Complexity of Minimizing Regularized Convex Quadratic Functions}, 
      author={Daniel Berg Thomsen and Nikita Doikov},
      year={2024},
      eprint={2404.17543},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2404.17543}, 
}
```