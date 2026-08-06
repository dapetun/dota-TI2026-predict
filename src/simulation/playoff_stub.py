"""Playoff bracket simulation — Stage 6 stub (not implemented).

Swiss group stage remains the production surface. A double-elim / TI main-event
bracket MC is roadmap-only; call sites must not import a missing module.
"""

from __future__ import annotations

from typing import Any


def simulate_playoffs_stub(
    *_args: Any,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Safe no-op playoff stub.

    Returns an empty result with an explicit ``implemented=False`` flag so UI /
    export can skip without ImportError.
    """
    return {
        "implemented": False,
        "reason": "Playoff bracket MC is roadmap Stage 6 — see docs/ROADMAP_v03.md",
        "teams": [],
        "champion_probs": {},
    }
