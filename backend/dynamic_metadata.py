# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dynamic metadata for ec-MLP."""

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    pass
else:
    pass

__all__ = ["dynamic_metadata"]


def __dir__() -> list[str]:
    return __all__


def dynamic_metadata(
    field: str,
    settings: dict[str, object] | None = None,
):
    """Get dynamic metadata for ec-MLP.

    Args:
        field: The metadata field to get
        settings: Additional settings

    Returns:
        The metadata value
    """
    if field == "version":
        # Read version from _version.py
        version_file = Path("src/ec_mlp/_version.py")
        if version_file.exists():
            with open(version_file) as f:
                exec(f.read())
                return locals().get("__version__", "1.0.0")
        return "1.0.0"
    else:
        raise ValueError(f"Unsupported field: {field}")
