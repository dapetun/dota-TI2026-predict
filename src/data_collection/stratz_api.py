"""STRATZ API client — DEPRECATED, not part of the production pipeline.

Broken GraphQL (typos in schema) and unused by ``train_compare`` / export.
Canonical data source: OpenDota via ``opendota_api`` + ``download_data`` /
``download_details``. Kept only so old scripts do not ImportError; do not call
from new code.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "src.data_collection.stratz_api is deprecated and excluded from the prod "
    "pipeline. Use OpenDota (opendota_api / download_details) instead.",
    DeprecationWarning,
    stacklevel=2,
)


class StratzClient:
    """Stub: STRATZ path removed from production surface."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "StratzClient is deprecated. Use OpenDotaClient / scripts/download_data.py."
        )


def download_stratz_data(*args, **kwargs):
    """Deprecated entry — always raises."""
    raise RuntimeError(
        "download_stratz_data is deprecated. Use scripts/download_data.py + download_details.py."
    )


if __name__ == "__main__":
    raise SystemExit("stratz_api is deprecated; use OpenDota downloaders")
