"""
CFBD data pull + dynamic tier assignment.

Set your key as an env var before running:
    export CFBD_API_KEY="your_key_here"

Or pass it directly to CFBDClient(api_key=...).
"""

import os
import cfbd
from cfbd.rest import ApiException

SEASONS = [2022, 2023, 2024, 2025]

# Recency weights applied at training time (most recent season weighted highest).
# Exponential decay, base 0.75 — tune later once we see backtest results.
SEASON_WEIGHTS = {season: 0.75 ** (max(SEASONS) - season) for season in SEASONS}


class CFBDClient:
    def __init__(self, api_key: str | None = None):
        api_key = api_key or os.environ.get("CFBD_API_KEY")
        if not api_key:
            raise ValueError(
                "No CFBD API key found. Set CFBD_API_KEY env var or pass api_key=..."
            )
        config = cfbd.Configuration(access_token=api_key)
        self.api_client = cfbd.ApiClient(config)
        self.games_api = cfbd.GamesApi(self.api_client)
        self.ratings_api = cfbd.RatingsApi(self.api_client)
        self.teams_api = cfbd.TeamsApi(self.api_client)
        self.rankings_api = cfbd.RankingsApi(self.api_client)
        self.metrics_api = cfbd.MetricsApi(self.api_client)
        self.betting_api = cfbd.BettingApi(self.api_client)

    def get_games(self, season: int) -> list:
        """FBS games for a season, regular season + postseason (excludes
        pure lower-division matchups like D2/D3 games that don't involve
        an FBS team)."""
        try:
            reg = self.games_api.get_games(year=season, season_type="regular", classification="fbs")
            post = self.games_api.get_games(year=season, season_type="postseason", classification="fbs")
            return reg + post
        except ApiException as e:
            print(f"Error fetching games for {season}: {e}")
            return []

    def get_sp_ratings(self, season: int) -> list:
        """SP+ ratings by team for a season."""
        try:
            return self.ratings_api.get_sp(year=season)
        except ApiException as e:
            print(f"Error fetching SP+ for {season}: {e}")
            return []

    def get_talent(self, season: int) -> list:
        """Team talent composite (recruiting-derived) for a season."""
        try:
            return self.teams_api.get_talent(year=season)
        except ApiException as e:
            print(f"Error fetching talent for {season}: {e}")
            return []

    def get_ap_rankings(self, season: int) -> list:
        """Weekly AP poll rankings for a season."""
        try:
            return self.rankings_api.get_rankings(year=season)
        except ApiException as e:
            print(f"Error fetching rankings for {season}: {e}")
            return []

    def get_team_ppa(self, season: int) -> list:
        """Predicted points added — team-level efficiency metric."""
        try:
            return self.metrics_api.get_predicted_points_added_by_team(year=season)
        except ApiException as e:
            print(f"Error fetching PPA for {season}: {e}")
            return []

    def get_betting_lines(self, season: int) -> list:
        """
        Betting lines (spread, over/under, moneylines) per game, one entry
        per sportsbook provider. NOTE: this is pulled for BENCHMARKING the
        model against the market, not as a training feature — see
        build_feature_table.py's lines_to_df() for how it's kept separate.
        """
        try:
            return self.betting_api.get_lines(year=season, season_type="regular") + \
                   self.betting_api.get_lines(year=season, season_type="postseason")
        except ApiException as e:
            print(f"Error fetching betting lines for {season}: {e}")
            return []


def build_dynamic_tiers(sp_ratings: list, week_ap_ranks: dict) -> dict:
    """
    Programmatic replacement for the workbook's manually-curated tier column.

    Tier 1: AP top-25 team, OR SP+ rating in roughly the top 25 nationally
    Tier 2: FBS team outside that band but with a positive/competitive SP+
    Tier 3: Weak FBS or any FCS opponent

    Returns {team_name: tier} for one season/week snapshot. This is a
    starting cutoff rule, not a re-derivation of Em's exact manual
    judgment calls — flag any teams that look mis-tiered once we backtest
    against the workbook's own 2025 output, and we can adjust the cutoff.
    """
    if not sp_ratings:
        return {}

    # Sort teams by SP+ overall rating, descending
    ranked = sorted(sp_ratings, key=lambda t: getattr(t, "rating", -999), reverse=True)
    tiers = {}
    for i, team in enumerate(ranked):
        name = team.team
        rank = i + 1
        ap_rank = week_ap_ranks.get(name)
        if ap_rank is not None and ap_rank <= 25:
            tiers[name] = 1
        elif rank <= 40:
            tiers[name] = 1
        elif rank <= 90:
            tiers[name] = 2
        else:
            tiers[name] = 3
    return tiers


if __name__ == "__main__":
    client = CFBDClient()  # reads CFBD_API_KEY from env
    for season in SEASONS:
        games = client.get_games(season)
        print(f"{season}: {len(games)} games pulled")