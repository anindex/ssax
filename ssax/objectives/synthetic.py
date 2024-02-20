import jax
import jax.numpy as jnp
from jax import jit
import numpy as np

from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from flax import struct

from .base import ObjectiveFn


@struct.dataclass
class Ackley(ObjectiveFn):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    optimal_value: float = 0.0
    a: float = 20
    b: float = 0.2
    c: float = 2 * jnp.pi

    @classmethod
    def create(
        cls,
        dim: int = 2,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: jax.Array = None,
        **kwargs: Any
    ) -> 'Ackley':
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        if bounds is None:
            bounds = jnp.array([(-6., 6.)]).repeat(dim, axis=0)
        optimizers = jnp.zeros((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        a, b, c = self.a, self.b, self.c
        part1 = -a * jnp.exp(-b / jnp.sqrt(self.dim) * jnp.linalg.norm(X, axis=-1))
        part2 = -(jnp.exp(jnp.mean(jnp.cos(c * X), axis=-1)))
        return part1 + part2 + a + jnp.e


@struct.dataclass
class Beale(ObjectiveFn):

    dim: int = 2
    optimal_value: float = 0.0
    optimizers: jax.Array = jnp.array([(3.0, 0.5)])

    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jax.Array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-4.5, 4.5), (-4.5, 4.5)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2**2) ** 2
        part3 = (2.625 - x1 + x1 * x2**3) ** 2
        return part1 + part2 + part3


@struct.dataclass
class Branin(ObjectiveFn):

    dim: int = 2
    optimal_value: float = 0.397887
    optimizers: jax.Array = jnp.array([(-jnp.pi, 12.275), (jnp.pi, 2.275), (9.42478, 2.475)])

    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jax.Array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-5.0, 10.0), (0.0, 15.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * jnp.pi**2) * X[..., 0] ** 2
            + 5 / jnp.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * jnp.pi)) * jnp.cos(X[..., 0])
        return t1**2 + t2 + 10


@struct.dataclass
class Bukin(ObjectiveFn):

    dim: int = 2
    optimal_value: float = 0.0
    optimizers: jnp.array = jnp.array([(-10.0, 1.0)])
    
    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-15.0, -5.0), (-3.0, 3.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        part1 = 100.0 * jnp.sqrt(jnp.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * jnp.abs(X[..., 0] + 10.0)
        return part1 + part2


@struct.dataclass
class Cosine8(ObjectiveFn):

    dim: int = 8
    optimal_value: float = 0.8
    optimizers: jnp.array = jnp.zeros(8)

    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-1.0, 1.0)]).repeat(8, axis=0)
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        return jnp.sum(0.1 * jnp.cos(5 * jnp.pi * X) - jnp.power(X, 2), axis=-1)


@struct.dataclass
class DropWave(ObjectiveFn):

    dim: int = 2
    optimal_value: float = -1.0
    optimizers: jnp.array = jnp.zeros(2)

    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-5.12, 5.12), (-5.12, 5.12)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        norm = jnp.linalg.norm(X, axis=-1) 
        part1 = 1.0 + jnp.cos(12.0 * norm)
        part2 = 0.5 * jnp.power(norm, 2) + 2.0
        return -part1 / part2


@struct.dataclass
class EggHolder(ObjectiveFn):

    dim: int = 2
    optimal_value: float = -959.6407
    optimizers: jnp.array = jnp.array([(512.0, 404.2319)])
    
    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-512.0, 512.0), (-512.0, 512.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = -(x2 + 47.0) * jnp.sin(jnp.sqrt(jnp.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * jnp.sin(jnp.sqrt(jnp.abs(x1 - (x2 + 47.0))))
        return part1 + part2
    

@struct.dataclass
class HolderTable(ObjectiveFn):

    dim: int = 2
    optimal_value: float = -19.2085
    optimizers: jnp.array = jnp.array([
        (8.05502, 9.66459),
        (-8.05502, -9.66459),
        (-8.05502, 9.66459),
        (8.05502, -9.66459),
    ])
    
    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-10.0, 10.0), (-10.0, 10.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        term = jnp.abs(1 - jnp.linalg.norm(X, axis=-1) / jnp.pi)
        return -(
            jnp.abs(jnp.sin(X[..., 0]) * jnp.cos(X[..., 1]) * jnp.exp(term))
        )


@struct.dataclass
class SixHumpCamel(ObjectiveFn):

    dim: int = 2
    optimal_value: float = -1.0316
    optimizers: jnp.array = jnp.array([(0.0898, -0.7126), (-0.0898, 0.7126)])
    
    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-3.0, 3.0), (-2.0, 2.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        x1, x2 = X[..., 0], X[..., 1]
        return (
            (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2
            + x1 * x2
            + (4 * x2**2 - 4) * x2**2
        )
    

@struct.dataclass
class ThreeHumpCamel(ObjectiveFn):

    dim: int = 2
    optimal_value: float = 0.0
    optimizers: jnp.array = jnp.array([(0.0, 0.0)])
    
    @classmethod
    def create(cls, 
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: jnp.array = None,
               **kwargs: Any):
        if bounds is None:
            bounds = jnp.array([(-5.0, 5.0), (-5.0, 5.0)])
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        x1, x2 = X[..., 0], X[..., 1]
        return 2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2


@struct.dataclass
class DixonPrice(ObjectiveFn):

    optimal_value: float = 0.0

    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'DixonPrice':
        if bounds is None:
            bounds = jnp.array([(-10.0, 10.0)]).repeat(dim, axis=0)
        optimizers = jnp.array([
            tuple(
                jnp.power(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, dim + 1)
            )
        ])
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        part1 = jnp.power(X[..., 0] - 1, 2)
        i = jnp.arange(2, self.dim + 1)
        part2 = jnp.sum(i * jnp.power(2.0 * jnp.power(X[..., 1:], 2) - X[..., :-1], 2), axis=-1)
        return part1 + part2


@struct.dataclass
class Griewank(ObjectiveFn):

    optimal_value: float = 0.0
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Griewank':
        if bounds is None:
            bounds = jnp.array([(-600.0, 600.0)]).repeat(dim, axis=0)
        optimizers = jnp.zeros((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        part1 = jnp.sum(X**2 / 4000.0, axis=-1)
        i = jnp.arange(1, self.dim + 1)
        part2 = -(jnp.prod(jnp.cos(X / jnp.sqrt(i)), axis=-1))
        return part1 + part2 + 1.0


@struct.dataclass
class Levy(ObjectiveFn):

    optimal_value: float = 0.0
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Levy':
        if bounds is None:
            bounds = jnp.array([(-10.0, 10.0)]).repeat(dim, axis=0)
        optimizers = jnp.ones((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
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
    

@struct.dataclass
class Michalewicz(ObjectiveFn):

    dim: int = 2  # NOTE: hard fixed dim = 2 for now
    optimal_value: float = -1.8013
    optimizers: jnp.array = jnp.array([(2.20290552, 1.57079633)])

    @classmethod
    def create(cls,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Michalewicz':
        if bounds is None:
            bounds = jnp.array([(0.0, jnp.pi)]).repeat(cls.dim, axis=0)
        return cls(noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        m = 10
        i = jnp.arange(1, self.dim + 1)
        return -(
            jnp.sum(
                jnp.sin(X) * jnp.sin(i * X**2 / jnp.pi) ** (2 * m), axis=-1
            )
        )
    

@struct.dataclass
class Powell(ObjectiveFn):

    optimal_value: float = 0.0
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Powell':
        if bounds is None:
            bounds = jnp.array([(-4.0, 5.0)]).repeat(dim, axis=0)
        optimizers = jnp.zeros((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        result = jnp.zeros_like(X[..., 0])
        for i in range(self.dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result
    

@struct.dataclass
class Rastrigin(ObjectiveFn):

    optimal_value: float = 0.0
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Rastrigin':
        if bounds is None:
            bounds = jnp.array([(-5.12, 5.12)]).repeat(dim, axis=0)
        optimizers = jnp.zeros((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        return 10.0 * self.dim + jnp.sum(
            X**2 - 10.0 * jnp.cos(2.0 * jnp.pi * X), axis=-1
        )
    

@struct.dataclass
class Rosenbrock(ObjectiveFn):

    optimal_value: float = 0.0
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'Rosenbrock':
        if bounds is None:
            bounds = jnp.array([(-5.0, 5.0)]).repeat(dim, axis=0)
        optimizers = jnp.ones((1, dim))
        return cls(dim=dim, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        return jnp.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            axis=-1,
        )


@struct.dataclass
class StyblinskiTang(ObjectiveFn):
    
    @classmethod
    def create(cls,
               dim: int = 2,
               noise_std: Optional[float] = None,
               negate: bool = False,
               bounds: Optional[List[Tuple[float, float]]] = None,
               **kwargs: Any
    ) -> 'StyblinskiTang':
        if bounds is None:
            bounds = jnp.array([(-5.0, 5.0)]).repeat(dim, axis=0)
        optimizers = jnp.array([tuple(-2.903534 for _ in range(dim))])
        optimal_value = -39.166166 * dim
        return cls(dim=dim, optimal_value=optimal_value, optimizers=optimizers, noise_std=noise_std, negate=negate, bounds=bounds, **kwargs)

    def evaluate(self, X: jax.Array) -> jax.Array:
        return 0.5 * (X**4 - 16 * X**2 + 5 * X).sum(axis=-1)
