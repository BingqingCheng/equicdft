"""Result-dictionary conventions shared by every solver path."""

from .._deprecation import AliasedOutputs
from ..model import OUTPUT_KEY_ALIASES

__all__ = ["RESULT_KEY_ALIASES", "aliased_result"]

# Deprecated result keys (equicdft 0.0.x) -> canonical keys; the old names
# stay as aliases of the same tensors until 0.2.0 and warn when read.
RESULT_KEY_ALIASES = dict(
    OUTPUT_KEY_ALIASES,
    equilibrium_chemical_potential="beta_mu",
)


def aliased_result(result):
    return AliasedOutputs.with_aliases(result, RESULT_KEY_ALIASES)
