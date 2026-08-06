"""OpenDota API client for match/hero/player data."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

# Cap recursive 429 retries so rate-limit storms cannot hang forever.
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_RETRY_BACKOFF_SEC: float = 60.0


class OpenDotaRateLimitError(RuntimeError):
    """Raised when OpenDota keeps returning HTTP 429 after max retries."""


class OpenDotaClient:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(
        self,
        rate_limit: float = 1.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_sec: float = DEFAULT_RETRY_BACKOFF_SEC,
    ):
        self.session = requests.Session()
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self._last_request = 0.0

    def _get(self, endpoint: str, params: dict | None = None, _attempt: int = 0) -> dict:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params, timeout=45)
        self._last_request = time.time()

        if resp.status_code == 429:
            if _attempt >= self.max_retries:
                raise OpenDotaRateLimitError(
                    f"OpenDota 429 after {self.max_retries} retries on {endpoint}"
                )
            wait = self.retry_backoff_sec * (1.0 + 0.25 * _attempt)
            print(f"Rate limited, waiting {wait:.0f}s (attempt {_attempt + 1}/{self.max_retries})...")
            time.sleep(wait)
            return self._get(endpoint, params, _attempt=_attempt + 1)

        resp.raise_for_status()
        return resp.json()

    def get_pro_matches(self, offset: int = 0, limit: int = 100) -> list[dict]:
        return self._get("/proMatches", {"offset": offset, "limit": limit})

    def get_match(self, match_id: int) -> dict:
        return self._get(f"/matches/{match_id}")

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
