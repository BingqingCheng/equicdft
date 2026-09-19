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


class AliasedOutputs(dict):
    """Output dictionary whose deprecated keys warn when they are read.

    Model and solver outputs carry their canonical keys and, until the old
    names are removed, the deprecated keys as aliases of the same tensors.
    Reading a deprecated key through ``[]`` or :meth:`get` warns once per
    call site; membership tests and iteration do not, so code that merely
    copies the dictionary stays silent.
    """

    __slots__ = ("_aliases",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aliases = {}

    @classmethod
    def with_aliases(cls, values, aliases):
        """Return ``values`` plus one alias entry per ``{old: new}`` pair present."""

        outputs = cls(values)
        outputs._aliases = dict(aliases)
        for old, new in aliases.items():
            if new in outputs:
                dict.__setitem__(outputs, old, dict.__getitem__(outputs, new))
        return outputs

    def _warn(self, key):
        new = self._aliases.get(key)
        if new is not None and dict.__contains__(self, key):
            warnings.warn(
                "output key '{}' is deprecated; use '{}' (the old key is "
                "removed in equicdft {})".format(key, new, REMOVAL_VERSION),
                EquicdftDeprecationWarning,
                stacklevel=3,
            )

    def __getitem__(self, key):
        self._warn(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._warn(key)
        return super().get(key, default)

    def copy(self):
        clone = type(self)(self)
        clone._aliases = dict(self._aliases)
        return clone

    def __reduce__(self):
        return (_rebuild_aliased_outputs, (dict(self), dict(self._aliases)))


def _rebuild_aliased_outputs(values, aliases):
    outputs = AliasedOutputs(values)
    outputs._aliases = aliases
    return outputs
