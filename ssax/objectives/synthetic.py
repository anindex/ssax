import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from typing import Any, Callable, Dict, Optional, Tuple, Union, List

from .base import ObjectiveFn


@jax.tree_util.register_pytree_node_class
class Ackley(ObjectiveFn):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        self.dim = dim
        if bounds is None:
            bounds = [(-6., 6.) for _ in range(dim)]
        self._optimizers = jnp.zeros(self.dim)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)
        self.a = 20
        self.b = 0.2
        self.c = 2 * jnp.pi

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        a, b, c = self.a, self.b, self.c
        part1 = -a * jnp.exp(-b / jnp.sqrt(self.dim) * jnp.linalg.norm(X, axis=-1))
        part2 = -(jnp.exp(jnp.mean(jnp.cos(c * X), axis=-1)))
        return part1 + part2 + a + jnp.e
    
    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class Beale(ObjectiveFn):

    _optimal_value = 0.0
    _optimizers = jnp.array([(3.0, 0.5)])

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2**2) ** 2
        part3 = (2.625 - x1 + x1 * x2**3) ** 2
        return part1 + part2 + part3


@jax.tree_util.register_pytree_node_class
class Branin(ObjectiveFn):

    _optimal_value = 0.397887
    _optimizers = [(-jnp.pi, 12.275), (jnp.pi, 2.275), (9.42478, 2.475)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-5.0, 10.0), (0.0, 15.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * jnp.pi**2) * X[..., 0] ** 2
            + 5 / jnp.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * jnp.pi)) * jnp.cos(X[..., 0])
        return t1**2 + t2 + 10


@jax.tree_util.register_pytree_node_class
class Bukin(ObjectiveFn):

    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-15.0, -5.0), (-3.0, 3.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        part1 = 100.0 * jnp.sqrt(jnp.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * jnp.abs(X[..., 0] + 10.0)
        return part1 + part2


@jax.tree_util.register_pytree_node_class
class Cosine8(ObjectiveFn):

    _optimal_value = 0.8
    _optimizers = [tuple(0.0 for _ in range(8))]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 8
        if bounds is None:
            bounds = [(-1.0, 1.0) for _ in range(8)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        return jnp.sum(0.1 * jnp.cos(5 * jnp.pi * X) - jnp.power(X, 2), axis=-1)


@jax.tree_util.register_pytree_node_class
class DropWave(ObjectiveFn):

    _optimal_value = -1.0
    _optimizers = [(0.0, 0.0)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        norm = jnp.linalg.norm(X, axis=-1) 
        part1 = 1.0 + jnp.cos(12.0 * norm)
        part2 = 0.5 * jnp.power(norm, 2) + 2.0
        return -part1 / part2


@jax.tree_util.register_pytree_node_class
class EggHolder(ObjectiveFn):

    _optimal_value = -959.6407
    _optimizers = [(512.0, 404.2319)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-512.0, 512.0), (-512.0, 512.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)
        
    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = -(x2 + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * jnp.sin(jnp.sqrt(jnp.abs(x1 - (x2 + 47.0))))
        return part1 + part2
    

@jax.tree_util.register_pytree_node_class
class HolderTable(ObjectiveFn):

    _optimal_value = -19.2085
    _optimizers = [
        (8.05502, 9.66459),
        (-8.05502, -9.66459),
        (-8.05502, 9.66459),
        (8.05502, -9.66459),
    ]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-10.0, 10.0), (-10.0, 10.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        term = jnp.abs(1 - jnp.linalg.norm(X, axis=-1) / jnp.pi)
        return -(
            jnp.abs(jnp.sin(X[..., 0]) * jnp.cos(X[..., 1]) * jnp.exp(term))
        )
    

@jax.tree_util.register_pytree_node_class
class SixHumpCamel(ObjectiveFn):

    _optimal_value = -1.0316
    _optimizers = [(0.0898, -0.7126), (-0.0898, 0.7126)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-3.0, 3.0), (-2.0, 2.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        x1, x2 = X[..., 0], X[..., 1]
        return (
            (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
            + x1 * x2
            + (4 * x2**2 - 4) * x2**2
        )
    

@jax.tree_util.register_pytree_node_class
class ThreeHumpCamel(ObjectiveFn):

    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]

    def __init__(self, 
                 noise_std: Optional[float] = None, 
                 negate: Optional[bool] = False, 
                 bounds: List[Tuple[float, float]] = None, 
                 **kwargs: Any):
        self.dim = 2
        if bounds is None:
            bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        super().__init__(noise_std, negate, bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        x1, x2 = X[..., 0], X[..., 1]
        return 2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2


@jax.tree_util.register_pytree_node_class
class DixonPrice(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [
            tuple(
                jnp.power(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.dim + 1)
            )
        ]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        d = self.dim
        part1 = jnp.power(X[..., 0] - 1, 2)
        i = jnp.arange(2, d + 1)
        part2 = jnp.sum(i * jnp.power(2.0 * jnp.power(X[..., 1:], 2) - X[..., :-1], 2), axis=-1)
        return part1 + part2

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
    

@jax.tree_util.register_pytree_node_class
class Griewank(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-600.0, 600.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        part1 = jnp.sum(X**2 / 4000.0, axis=-1)
        i = jnp.arange(1, self.dim + 1)
        part2 = -(jnp.prod(jnp.cos(X / jnp.sqrt(i)), axis=-1))
        return part1 + part2 + 1.0

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class Levy(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 6,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        w = 1.0 + (X - 1.0) / 4.0
        part1 = jnp.sin(jnp.pi * w[..., 0]) ** 2
        part2 = jnp.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * jnp.sin(jnp.pi * w[..., :-1] + 1.0) ** 2),
            axis=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + jnp.sin(2.0 * jnp.pi * w[..., -1]) ** 2
        )
        return part1 + part2 + part3

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
    

@jax.tree_util.register_pytree_node_class
class Michalewicz(ObjectiveFn):

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(0.0, jnp.pi) for _ in range(self.dim)]
        optvals = {2: -1.80130341, 5: -4.687658, 10: -9.66015}
        optimizers = {2: [(2.20290552, 1.57079633)]}
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)
        self.i = jnp.arange(1, self.dim + 1)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        m = 10
        return -(
            jnp.sum(
                jnp.sin(X) * jnp.sin(self.i * X**2 / jnp.pi) ** (2 * m), axis=-1
            )
        )

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
    

@jax.tree_util.register_pytree_node_class
class Powell(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-4.0, 5.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        result = jnp.zeros_like(X[..., 0])
        for i in range(self.dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
    

@jax.tree_util.register_pytree_node_class
class Rastrigin(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        return 10.0 * self.dim + jnp.sum(
            X**2 - 10.0 * jnp.cos(2.0 * jnp.pi * X), axis=-1
        )

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
    

@jax.tree_util.register_pytree_node_class
class Rosenbrock(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        return jnp.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            axis=-1,
        )

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)


@jax.tree_util.register_pytree_node_class
class StyblinskiTang(ObjectiveFn):

    _optimal_value = 0.0

    def __init__(
        self,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
        **kwargs: Any
    ) -> None:
        self.dim = dim
        if bounds is None:
            bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self._optimal_value = -39.166166 * self.dim
        self._optimizers = [tuple(-2.903534 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    @jit
    def evaluate(self, X: jnp.array) -> jnp.array:
        return 0.5 * (X**4 - 16 * X**2 + 5 * X).sum(axis=-1)

    def tree_flatten(self):  
        return (), {
            "dim": self.dim,
            "noise_std": self.noise_std,
            "negate": self.negate,
            "bounds": self.bounds,
        }

    @classmethod
    def tree_unflatten(cls, aux_data, children):  
        return cls(*children, **aux_data)
