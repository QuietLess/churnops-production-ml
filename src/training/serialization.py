"""Safe model serialization helpers.

MLflow saves sklearn models with skops, which refuses to load types that were not
explicitly trusted. Instead of trusting everything, we compute the types the model
needs and only accept them if they come from an allowlist of known libraries plus
this project's own `src.` package.
"""

from __future__ import annotations

import skops.io as sio

ALLOWED_TYPE_PREFIXES = (
    "sklearn.",
    "lightgbm.",
    "numpy.",
    "collections.",
    "builtins.",
    "src.",
)


class UntrustedModelTypeError(RuntimeError):
    pass


def trusted_types_for(model) -> list[str]:
    needed = sio.get_untrusted_types(data=sio.dumps(model))
    rejected = [t for t in needed if not t.startswith(ALLOWED_TYPE_PREFIXES)]
    if rejected:
        raise UntrustedModelTypeError(f"Model references non-allowlisted types: {rejected}")
    return sorted(needed)
