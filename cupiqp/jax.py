"""JAX interface of cuPIQP: ``cupiqp.jax``.

Importing this module imports JAX. ``import cupiqp`` alone does not; the
attribute ``cupiqp.jax`` loads this module on first access. See
:mod:`cupiqp.interfaces.jax` for the implementation.
"""

from .interfaces.jax import *  # noqa: F401,F403
from .interfaces.jax import __all__  # noqa: F401
