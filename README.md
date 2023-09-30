# Sinkhorn Step in Jax (ssax)

This `ssax` repository demonstrates the proof of concept for the Sinkhorn Step - a batch gradient-free optimizer in Jax. `ssax` is heavily inspired by the code structure of [OTT-JAX](https://github.com/ott-jax/ott) to utilize most of its linear solvers, enabling the users to easily switch between different solver flavors.


## Installation

Simply activate your conda/Python environment and run

```azure
pip install -e .
```

## Run some experiments

Run

```azure
python scripts/run.py
```

and find result animations in the `logs/` folder.

## Citation

If you found this work useful, please consider citing this reference:

```
@inproceedings{le2023accelerating,
  title={Accelerating Motion Planning via Optimal Transport},
  author={Le, An T. and Chalvatzaki, Georgia and Biess, Armin and Peters, Jan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

## See also

The [OTT-JAX documentation](https://ott-jax.readthedocs.io/en/latest/) for more details on the linear solvers.
