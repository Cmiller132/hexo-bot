"""hexfield — variable-geometry hex-lattice model package.

The model domain is the support set (stones ∪ full legal set ∪ 1-ring halo);
every engine-legal cell carries a policy logit.

Exposes the submodules: constants, geometry, support, features.
"""

from . import constants, geometry, support, features

__all__ = ["constants", "geometry", "support", "features"]
