"""Deprecation helpers shared by every compatibility shim in the package.

Every shim funnels through this module so that removing them in the next
minor version is a search for its name. Warnings use
:class:`EquicdftDeprecationWarning`, a ``DeprecationWarning`` subclass, so
users can silence or escalate them separately from other libraries'
deprecations (``warnings.simplefilter("error", EquicdftDeprecationWarning)``).
"""

import functools
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


def keyword_only(n_positional, legacy_order):
    """Accept the 0.0.x positional call form of a now keyword-only signature.

    Decorates ``__init__`` (or any method): calls that pass more than
    ``n_positional`` arguments after ``self`` are mapped onto the parameter
    names in ``legacy_order``, the positional order the signature had before
    it became keyword-only, and warn once per call site. The decorated
    signature itself is free to order its keyword-only parameters as it
    likes. Removed in :data:`REMOVAL_VERSION`, after which the extra
    positionals raise ``TypeError`` from the signature itself.
    """

    legacy_order = tuple(legacy_order)

    def decorate(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            if len(args) > n_positional:
                extra = args[n_positional:]
                names = legacy_order[n_positional:n_positional + len(extra)]
                owner = type(self).__name__
                if len(names) < len(extra):
                    raise TypeError(
                        "{}() takes at most {} positional arguments ({} given)".format(
                            owner, len(legacy_order), len(args),
                        )
                    )
                for name in names:
                    if name in kwargs:
                        raise TypeError(
                            "{}() got multiple values for argument '{}'".format(owner, name)
                        )
                warnings.warn(
                    "{}() takes {} positional argument{}; pass {} by keyword "
                    "(positional use is removed in equicdft {})".format(
                        owner, n_positional, "" if n_positional == 1 else "s",
                        ", ".join(names), REMOVAL_VERSION,
                    ),
                    EquicdftDeprecationWarning,
                    stacklevel=2,
                )
                kwargs = dict(kwargs, **dict(zip(names, extra)))
                args = args[:n_positional]
            return method(self, *args, **kwargs)

        return wrapper

    return decorate
