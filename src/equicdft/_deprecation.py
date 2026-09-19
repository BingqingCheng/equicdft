"""Deprecation helpers shared by every compatibility shim in the package.

Every shim funnels through this module so that removing them in the next
minor version is a search for its name. Warnings use
:class:`EquicdftDeprecationWarning`, a ``DeprecationWarning`` subclass, so
users can silence or escalate them separately from other libraries'
deprecations (``warnings.simplefilter("error", EquicdftDeprecationWarning)``).
"""

import warnings


REMOVAL_VERSION = "0.2.0"


class EquicdftDeprecationWarning(DeprecationWarning):
    """A deprecated equicdft import path, keyword, or output key was used."""


def warn_moved_module(old_name: str, new_name: str) -> None:
    """Warn that ``old_name`` is a shim for ``new_name``.

    Called at import time of the shim module; ``stacklevel`` points at the
    statement that imported it.
    """

    warnings.warn(
        "{} moved to {}; the old import path is removed in equicdft {}".format(
            old_name, new_name, REMOVAL_VERSION,
        ),
        EquicdftDeprecationWarning,
        stacklevel=3,
    )
