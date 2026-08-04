"""STRATZ API client for collecting pro match data."""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class StratzClient:
    BASE_URL = "https://api.stratz.com/graphql"

    def __init__(self, token: str = None):
        self.token = token or os.environ.get("STRATZ_TOKEN", "")
        if not self.token:
            raise ValueError("STRATZ token required. Set STRATZ_TOKEN env var.")
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _query(self, query: str, variables: dict = None) -> dict:
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self.session.post(self.BASE_URL, json=payload)
        resp.raise_for_status()
        time.sleep(0.3)
        return resp.json()

    def get_pro_matches(
        self,
        limit: int = 100,
        offset: int = 0,
        min_duration: int = 600,
        league_ids: list[int] = None,
    ) -> list[dict]:
        """Fetch recent pro matches."""
        league_filter = ""
        if league_ids:
            ids_str = ",".join(str(lid) for lid in league_ids)
            league_filter = f", leagueId: [{ids_str}]"

        query = """
        query GetProMatches($take: Int, $skip: Int) {
            matches(
                take: $take
                skip: $skip
                orderBy: DESC
                type: MATCH_TYPEköy_NORMAL
                minDuration: %d
                isParsed: true
                %s
            ) {
                id
                didRadiantWin
                durationSeconds
                startDateTime
                gameMode
                leagueId
                tournamentId
                radiantTeamId
                direTeamId
                players {
                    heroId
                    steamAccountId
                    isRadiant
                    networth
                    kills
                    deaths
                    assists
                    goldPerMinute
                    experiencePerMinute
                    heroDamage
                    towerDamage
                    healing
                    aghanimsScepter
                    aghanimsShard
                    is Câm
                    level
                }
            }
        }
        """ % (min_duration, league_filter)

        result = self._query(query, {"take": limit, "skip": offset})
        return result.get("data", {}).get("matches", [])

    def get_team_matches(
        self, team_id: int, limit: int = 100
    ) -> list[dict]:
        """Fetch matches for a specific team."""
        query = """
        query GetTeamMatches($teamId: Long!, $take: Int) {
            team(id: $teamId) {
                name
                matches(request: {take: $take, orderBy: DESC, isParsed: true}) {
                    id
                    didRadiantWin
                    durationSeconds
                    startDateTime
                    leagueId
                    radiantTeamId
                    direTeamId
                    players {
                        heroId
                        steamAccountId
                        isRadiant
                        networth
                        kills
                        deaths
                        assists
                        goldPerMinute
                        experiencePerMinute
                        heroDamage
                        towerDamage
                        level
                    }
                }
            }
        }
        """
        result = self._query(query, {"teamId": team_id, "take": limit})
        team_data = result.get("data", {}).get("team", {})
        return team_data.get("matches", []) if team_data else []

    def get_hero_stats(self) -> list[dict]:
        """Fetch current hero statistics."""
        query = """
        {
            heroes {
                id
                displayName
                shortName
                proWin
                proPick
                proBan
            }
        }
        """
        result = self._query(query)
        return result.get("data", {}).get("heroes", [])

    def get_league_list(
        self, tier: int = 1, limit: int = 50
    ) -> list[dict]:
        """Fetch pro leagues/tournaments."""
        query = """
        query GetLeagues($tier: LeagueTier, $take: Int) {
            leagues(request: {tier: $tier, take: $take, orderBy: DESC}) {
                id
                name
                tier
                region
                startTime
                prizePool
            }
        }
        """
        result = self._query(query, {"tier": tier, "take": limit})
        return result.get("data", {}).get("leagues", [])

    def download_all_pro_matches(
        self,
        output_dir: str,
        start_date: str = "2024-01-01",
        batch_size: int = 100,
        max_matches: int = 50000,
    ) -> str:
        """Download all pro matches since start_date in batches."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        all_matches = []
        offset = 0

        while offset < max_matches:
            print(f"Fetching matches offset={offset}...")
            matches = self.get_pro_matches(limit=batch_size, offset=offset)
            if not matches:
                break

            for m in matches:
                if m.get("startDateTime", 0) >= start_ts:
                    all_matches.append(m)

            offset += batch_size

            if len(matches) < batch_size:
                break

        df = pd.json_normalize(all_matches, max_level=1)
        output_file = output_path / "stratz_pro_matches.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} matches to {output_file}")
        return str(output_file)


def download_stratz_data(
    token: str = None,
    output_dir: str = "data/raw",
    start_date: str = "2024-01-01",
):
    """Main entry point for STRATZ data collection."""
    client = StratzClient(token)
    return client.download_all_pro_matches(output_dir, start_date)


if __name__ == "__main__":
    download_stratz_data()
