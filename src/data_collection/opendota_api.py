"""OpenDota API client for match/hero/player data."""

import os
import time
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Optional


class OpenDotaClient:
    BASE_URL = "https://api.opendota.com/api"

    def __init__(self, rate_limit: float = 1.0):
        self.session = requests.Session()
        self.rate_limit = rate_limit
        self._last_request = 0

    def _get(self, endpoint: str, params: dict = None) -> dict:
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        resp = self.session.get(url, params=params)
        self._last_request = time.time()

        if resp.status_code == 429:
            print("Rate limited, waiting 60s...")
            time.sleep(60)
            return self._get(endpoint, params)

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
