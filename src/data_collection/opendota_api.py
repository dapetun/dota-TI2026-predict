"""OpenDota API client for match/hero/player data."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

# Cap recursive 429 retries so rate-limit storms cannot hang forever.
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_RETRY_BACKOFF_SEC: float = 60.0
# /matches/{id} often hangs on large parsed payloads; explorer is ~0.5s.
DEFAULT_TIMEOUT_SEC: float = 90.0
DEFAULT_MATCH_REST_TIMEOUT_SEC: float = 20.0


class OpenDotaRateLimitError(RuntimeError):
    """Raised when OpenDota keeps returning HTTP 429 after max retries."""


def explorer_rows_to_match_detail(rows: list[dict], match_id: int) -> dict:
    """Build an OpenDota-shaped match dict from explorer JOIN rows."""
    if not rows:
        raise ValueError(f"explorer returned no rows for match_id={match_id}")

    head = rows[0]
    players: list[dict] = []
    for row in rows:
        slot = int(row.get("player_slot") or 0)
        players.append(
            {
                "account_id": row.get("account_id"),
                "hero_id": int(row.get("hero_id") or 0),
                "player_slot": slot,
                "isRadiant": slot < 128,
                "is_radiant": slot < 128,
                "kills": int(row.get("kills") or 0),
                "deaths": int(row.get("deaths") or 0),
                "assists": int(row.get("assists") or 0),
                "gold_per_min": int(row.get("gold_per_min") or 0),
                "xp_per_min": int(row.get("xp_per_min") or 0),
                "hero_damage": int(row.get("hero_damage") or 0),
                "tower_damage": int(row.get("tower_damage") or 0),
                "healing": int(row.get("hero_healing") or 0),
                "level": int(row.get("level") or 0),
                "lane_role": int(row.get("lane_role") or 0),
            }
        )

    return {
        "match_id": int(head.get("match_id") or match_id),
        "start_time": int(head.get("start_time") or 0),
        "duration": int(head.get("duration") or 0),
        "radiant_win": bool(head.get("radiant_win")),
        "radiant_team_id": head.get("radiant_team_id"),
        "dire_team_id": head.get("dire_team_id"),
        "leagueid": head.get("leagueid"),
        "players": players,
        "_source": "opendota_explorer",
    }


class OpenDotaClient:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(
        self,
        rate_limit: float = 1.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
        timeout: float = DEFAULT_TIMEOUT_SEC,
        api_key: str | None = None,
    ):
        self.session = requests.Session()
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.timeout = timeout
        self.api_key = api_key if api_key is not None else os.environ.get("OPENDOTA_API_KEY")
        self._last_request = 0.0

    def _params(self, params: dict | None = None) -> dict:
        """Merge caller params with optional API key."""
        out = dict(params or {})
        if self.api_key and "api_key" not in out:
            out["api_key"] = self.api_key
        return out

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        _attempt: int = 0,
        timeout: float | None = None,
    ) -> dict:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        resp = self.session.get(
            url,
            params=self._params(params),
            timeout=self.timeout if timeout is None else timeout,
        )
        self._last_request = time.time()

        if resp.status_code == 429:
            if _attempt >= self.max_retries:
                raise OpenDotaRateLimitError(
                    f"OpenDota 429 after {self.max_retries} retries on {endpoint}"
                )
            wait = self.retry_backoff_sec * (1.0 + 0.25 * _attempt)
            print(f"Rate limited, waiting {wait:.0f}s (attempt {_attempt + 1}/{self.max_retries})...")
            time.sleep(wait)
            return self._get(endpoint, params, _attempt=_attempt + 1, timeout=timeout)

        resp.raise_for_status()
        return resp.json()

    def get_pro_matches(self, offset: int = 0, limit: int = 100) -> list[dict]:
        return self._get("/proMatches", {"offset": offset, "limit": limit})

    def get_match(
        self,
        match_id: int,
        *,
        timeout: float | None = None,
    ) -> dict:
        """GET /matches/{id} (large payload; may hang under OpenDota load)."""
        return self._get(
            f"/matches/{match_id}",
            timeout=DEFAULT_MATCH_REST_TIMEOUT_SEC if timeout is None else timeout,
        )

    def get_match_explorer(self, match_id: int) -> dict:
        """Fetch match+players via /explorer SQL (fast fallback when /matches hangs)."""
        sql = f"""
SELECT pm.match_id, pm.account_id, pm.player_slot, pm.hero_id, pm.kills, pm.deaths, pm.assists,
       pm.gold_per_min, pm.xp_per_min, pm.hero_damage, pm.tower_damage, pm.hero_healing, pm.level,
       m.start_time, m.duration, m.radiant_win, m.radiant_team_id, m.dire_team_id, m.leagueid
FROM player_matches pm
JOIN matches m ON m.match_id = pm.match_id
WHERE pm.match_id = {int(match_id)}
""".strip()
        payload = self._get("/explorer", {"sql": sql}, timeout=45.0)
        if isinstance(payload, dict) and payload.get("err"):
            raise RuntimeError(f"OpenDota explorer error: {payload['err']}")
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected explorer payload for match_id={match_id}")
        return explorer_rows_to_match_detail(rows, int(match_id))

    def get_match_resilient(
        self,
        match_id: int,
        *,
        source: str = "explorer",
        rest_timeout: float = DEFAULT_MATCH_REST_TIMEOUT_SEC,
    ) -> dict:
        """Fetch match detail: explorer (default), REST, or REST→explorer on timeout.

        ``source``: ``explorer`` | ``rest`` | ``auto`` (REST first, explorer fallback).
        """
        mode = (source or "explorer").lower().strip()
        if mode == "explorer":
            return self.get_match_explorer(match_id)
        if mode == "rest":
            return self.get_match(match_id, timeout=rest_timeout)
        # auto: try REST briefly, then explorer
        try:
            return self.get_match(match_id, timeout=rest_timeout)
        except Exception:
            return self.get_match_explorer(match_id)

    def get_heroes(self) -> list[dict]:
        return self._get("/heroes")

    def get_hero_stats(self, hero_id: int) -> dict:
        return self._get(f"/heroes/{hero_id}")

    def get_teams(self) -> list[dict]:
        return self._get("/teams")

    def get_team_matches(self, team_id: int) -> list[dict]:
        return self._get(f"/teams/{team_id}/matches")

    def get_team_players(self, team_id: int) -> list[dict]:
        return self._get(f"/teams/{team_id}/players")

    def download_pro_matches(
        self,
        output_dir: str,
        max_matches: int = 50000,
        batch_size: int = 100,
    ) -> str:
        """Download all available pro matches."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_matches = []
        offset = 0

        while offset < max_matches:
            print(f"Fetching pro matches offset={offset}...")
            matches = self.get_pro_matches(offset=offset, limit=batch_size)
            if not matches:
                break
            all_matches.extend(matches)
            offset += batch_size

            if len(matches) < batch_size:
                break

        df = pd.json_normalize(all_matches, max_level=1)
        output_file = output_path / "opendota_pro_matches.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} matches to {output_file}")
        return str(output_file)

    def download_hero_data(self, output_dir: str) -> str:
        """Download hero metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        heroes = self.get_heroes()
        df = pd.DataFrame(heroes)
        output_file = output_path / "heroes.json"
        df.to_json(output_file, orient="records", indent=2)
        print(f"Saved {len(heroes)} heroes to {output_file}")
        return str(output_file)

    def download_team_data(self, team_ids: list[int], output_dir: str) -> str:
        """Download match history for specific teams."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_data = {}
        for tid in team_ids:
            print(f"Fetching team {tid}...")
            matches = self.get_team_matches(tid)
            all_data[tid] = matches

        output_file = output_path / "team_matches.json"
        with open(output_file, "w") as f:
            json.dump(all_data, f)
        print(f"Saved data for {len(team_ids)} teams to {output_file}")
        return str(output_file)


def download_opendota_data(output_dir: str = "data/raw"):
    """Main entry point."""
    client = OpenDotaClient()
    client.download_pro_matches(output_dir)
    client.download_hero_data(output_dir)
    return output_dir


if __name__ == "__main__":
    download_opendota_data()
