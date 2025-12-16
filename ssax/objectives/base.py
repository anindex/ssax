import abc
from typing import Any, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import struct

__all__ = ["ObjectiveFn"]


@struct.dataclass
class ObjectiveFn(abc.ABC):
    """Base abstract class for optimization objectives.
    
    Attributes:
        dim: Dimensionality of the problem (Static).
        bounds: Search space bounds of shape [dim, 2] (Dynamic Array).
        optimizers: Known global optimizer locations [num_opts, dim] (Dynamic Array).
        optimal_value: The known global optimum value (Static Float, for benchmarking).
        noise_std: Standard deviation of additive Gaussian noise (Static Float).
        negate: If True, the function output is negated (Static Bool). 
                Useful for turning maximization into minimization.
    """

    dim: int = struct.field(pytree_node=False)
    
    bounds: Optional[jax.Array] = struct.field(default=None, pytree_node=True)
    optimizers: Optional[jax.Array] = struct.field(default=None, pytree_node=True)
    
    # Metadata / Scalar configs can be static
    optimal_value: Optional[float] = struct.field(default=None, pytree_node=False)
    noise_std: Optional[float] = struct.field(default=None, pytree_node=False)
    negate: bool = struct.field(default=False, pytree_node=False)

    @abc.abstractmethod
    def evaluate(self, X: jax.Array) -> jax.Array:
        """Compute the raw objective value.

        Args:
          X: Input array of shape [..., dim].

        Returns:
          The cost array of shape [...].
        """
        pass

    def __call__(self, X: jax.Array) -> jax.Array:
        """Compute the final cost, applying negation if configured.
        
        Args:
            X: Input points [..., dim]
            
        Returns:
            Cost values [...]. 
            If negate=True, returns -evaluate(X).
        """
        cost = self.evaluate(X)

        if self.negate:
            return -cost
        return cost