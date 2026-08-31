"""
Builds the final training table: one row per game, with pre-game features
for both teams (SP+, talent, PPA, power rating, SOS) so the model predicts
using only information available BEFORE kickoff — no leakage.

Usage:
    export CFBD_API_KEY="your_key_here"
    python features/build_feature_table.py
"""

import os
import pandas as pd
from cfbd_pull import CFBDClient, SEASONS, SEASON_WEIGHTS, build_dynamic_tiers
from power_rating import GameResult, Locale, score_game, TeamSeasonPower


def games_to_df(games: list) -> pd.DataFrame:
    rows = []
    for g in games:
        if g.home_points is None or g.away_points is None:
            continue  # game not yet played, skip
        rows.append({
            "game_id": g.id,
            "season": g.season,
            "week": g.week,
            "season_type": g.season_type,
            "start_date": g.start_date,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "home_points": g.home_points,
            "away_points": g.away_points,
            "neutral_site": g.neutral_site,
            "conference_game": g.conference_game,
        })
    return pd.DataFrame(rows)


def sp_to_df(sp_ratings: list, season: int) -> pd.DataFrame:
    rows = []
    for t in sp_ratings:
        if t.year != season:
            continue
        rows.append({
            "season": season,
            "team": t.team,
            "sp_rating": getattr(t, "rating", None),
            "sp_offense": getattr(t.offense, "rating", None) if getattr(t, "offense", None) else None,
            "sp_defense": getattr(t.defense, "rating", None) if getattr(t, "defense", None) else None,
        })
    return pd.DataFrame(rows)


def talent_to_df(talent: list, season: int) -> pd.DataFrame:
    rows = [{"season": season, "team": t.team, "talent": t.talent} for t in talent if t.year == season]
    return pd.DataFrame(rows)


def lines_to_df(betting_games: list) -> pd.DataFrame:
    """
    One row per game_id with vegas_spread / vegas_over_under.

    NOTE ON SIGN CONVENTION: CFBD's spread follows standard sportsbook
    convention — negative means the HOME team is favored (e.g. -7.0 means
    home favored by 7). This matches our `margin` column's sign
    (home_points - away_points), so a model beating Vegas means its
    predicted margin is closer to actual margin than -vegas_spread is.

    Prefers the "consensus" provider when present; otherwise averages
    across whichever books reported a line for that game. BENCHMARKING
    USE ONLY — this table is merged onto the training data as separate
    vegas_* columns, deliberately excluded from the model's feature list
    (see NON_FEATURE_COLUMNS in the training script) so the model can't
    just learn to copy the market instead of predicting on its own.
    """
    rows = []
    for g in betting_games:
        if not g.lines:
            continue
        consensus = [l for l in g.lines if l.provider.lower() == "consensus"]
        source_lines = consensus if consensus else g.lines

        spreads = [l.spread for l in source_lines if l.spread is not None]
        totals = [l.over_under for l in source_lines if l.over_under is not None]

        if not spreads and not totals:
            continue

        rows.append({
            "game_id": g.id,
            "vegas_spread": sum(spreads) / len(spreads) if spreads else None,
            "vegas_over_under": sum(totals) / len(totals) if totals else None,
            "vegas_num_books": len(source_lines),
        })
    return pd.DataFrame(rows)


def ap_ranks_by_week(rankings: list) -> dict:
    """
    Returns {(season, week): {team: ap_rank}} from CFBD rankings response.
    Uses the AP poll specifically.
    """
    out = {}
    for week_data in rankings:
        season = week_data.season
        week = week_data.week
        for poll in week_data.polls:
            # NOTE: exact poll name string wasn't verifiable without a live
            # API call (no key was available while building this). Matching
            # loosely on "ap" to catch "AP Top 25" or similar variants —
            # print poll.poll the first time you run this for real and
            # tighten to an exact match once confirmed.
            if "ap" not in poll.poll.lower():
                continue
            ranks = {r.school: r.rank for r in poll.ranks}
            out[(season, week)] = ranks
    return out


def locale_for_row(row, team_is_home: bool) -> Locale:
    if row["neutral_site"]:
        return Locale.NEUTRAL
    return Locale.HOME if team_is_home else Locale.AWAY


def build_power_ratings_for_season(games_df: pd.DataFrame, tiers_by_week: dict, ap_ranks_by_week_map: dict, season: int) -> dict:
    """
    Walks a season's games in TRUE chronological order per team, computing
    running power rating BEFORE each game (so it can be joined back as a
    pre-game feature with no leakage).

    IMPORTANT: sorts by start_date ONLY, not by "week". CFBD's postseason
    week numbering resets independently from the regular season (e.g. a
    January bowl game can carry week=1, identical to that team's actual
    season opener) — sorting by week would process that bowl game before
    the team's real mid-season games, corrupting every snapshot in between.

    Snapshots are keyed by game_id (not week) since week numbers can
    collide between regular season and postseason for the same team —
    game_id is guaranteed unique per matchup, so there's no risk of one
    game's snapshot silently overwriting another's.

    Returns {(team, game_id): power rating BEFORE this specific game}
    """
    season_games = games_df[games_df["season"] == season].sort_values("start_date")
    team_running = {}  # team -> TeamSeasonPower
    snapshots = {}      # (team, game_id) -> power rating BEFORE this game

    for _, row in season_games.iterrows():
        week = row["week"]
        tiers = tiers_by_week.get((season, week), {})
        ap_ranks = ap_ranks_by_week_map.get((season, week), {})
        game_id = row["game_id"]

        for team_col, opp_col, is_home in [("home_team", "away_team", True), ("away_team", "home_team", False)]:
            team = row[team_col]
            opp = row[opp_col]

            # snapshot BEFORE this game = current accumulated state
            prior = team_running.get(team, TeamSeasonPower(team=team))
            snapshots[(team, game_id)] = prior.power_rating  # pre-game value

            opp_tier = tiers.get(opp, 3)  # default weakest tier if unknown (e.g. FCS)
            opp_ap_rank = ap_ranks.get(opp)
            locale = locale_for_row(row, is_home)
            pts_for = row["home_points"] if is_home else row["away_points"]
            pts_against = row["away_points"] if is_home else row["home_points"]

            game_result = GameResult(
                opponent=opp, opponent_tier=opp_tier, opponent_ap_rank=opp_ap_rank,
                locale=locale, pts_for=pts_for, pts_against=pts_against,
            )
            gs = score_game(game_result)

            updated = TeamSeasonPower(team=team, games=list(prior.games) + [gs])
            team_running[team] = updated

    return snapshots


def build_full_table(api_key: str) -> pd.DataFrame:
    client = CFBDClient(api_key=api_key)

    all_games = []
    all_sp = []
    all_talent = []
    all_rankings = []
    all_lines = []

    for season in SEASONS:
        print(f"Pulling {season}...")
        all_games.extend(client.get_games(season))
        all_sp.extend(client.get_sp_ratings(season))
        all_talent.extend(client.get_talent(season))
        all_rankings.extend(client.get_ap_rankings(season))
        all_lines.extend(client.get_betting_lines(season))

    games_df = games_to_df(all_games)
    sp_df = pd.concat([sp_to_df(all_sp, s) for s in SEASONS], ignore_index=True) if all_sp else pd.DataFrame()
    talent_df = pd.concat([talent_to_df(all_talent, s) for s in SEASONS], ignore_index=True) if all_talent else pd.DataFrame()
    ap_ranks_map = ap_ranks_by_week(all_rankings)

    # Build dynamic tiers per season (SP+ + AP rank based cutoffs)
    tiers_by_week = {}
    for season in SEASONS:
        season_sp = [t for t in all_sp if t.year == season]
        weeks_in_season = games_df[games_df["season"] == season]["week"].dropna().unique()
        for week in weeks_in_season:
            ap_this_week = ap_ranks_map.get((season, week), {})
            tiers_by_week[(season, week)] = build_dynamic_tiers(season_sp, ap_this_week)

    # Power ratings, computed pre-game per team per specific game (keyed by
    # game_id, not week — see build_power_ratings_for_season docstring)
    power_snapshots = {}
    for season in SEASONS:
        power_snapshots.update(
            {(season, k[0], k[1]): v for k, v in build_power_ratings_for_season(
                games_df, tiers_by_week, ap_ranks_map, season
            ).items()}
        )

    # Attach recency weight
    games_df["sample_weight"] = games_df["season"].map(SEASON_WEIGHTS)

    # Join power ratings, SP+, talent onto each game row for both teams
    def get_power(season, team, game_id):
        return power_snapshots.get((season, team, game_id), 0.0)

    games_df["home_power_rating"] = games_df.apply(
        lambda r: get_power(r["season"], r["home_team"], r["game_id"]), axis=1
    )
    games_df["away_power_rating"] = games_df.apply(
        lambda r: get_power(r["season"], r["away_team"], r["game_id"]), axis=1
    )

    if not sp_df.empty:
        games_df = games_df.merge(
            sp_df.rename(columns={"team": "home_team", "sp_rating": "home_sp_rating",
                                   "sp_offense": "home_sp_offense", "sp_defense": "home_sp_defense"}),
            on=["season", "home_team"], how="left",
        )
        games_df = games_df.merge(
            sp_df.rename(columns={"team": "away_team", "sp_rating": "away_sp_rating",
                                   "sp_offense": "away_sp_offense", "sp_defense": "away_sp_defense"}),
            on=["season", "away_team"], how="left",
        )

    if not talent_df.empty:
        games_df = games_df.merge(
            talent_df.rename(columns={"team": "home_team", "talent": "home_talent"}),
            on=["season", "home_team"], how="left",
        )
        games_df = games_df.merge(
            talent_df.rename(columns={"team": "away_team", "talent": "away_talent"}),
            on=["season", "away_team"], how="left",
        )

    # Targets
    games_df["margin"] = games_df["home_points"] - games_df["away_points"]
    games_df["total_points"] = games_df["home_points"] + games_df["away_points"]
    games_df["home_win"] = (games_df["margin"] > 0).astype(int)

    # Betting lines — BENCHMARK ONLY, not a training feature (see lines_to_df
    # docstring). Merged last, by game_id, so it never collides with the
    # team-keyed merges above.
    lines_df = lines_to_df(all_lines)
    if not lines_df.empty:
        games_df = games_df.merge(lines_df, on="game_id", how="left")

    return games_df


if __name__ == "__main__":
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        raise SystemExit("Set CFBD_API_KEY environment variable first.")

    df = build_full_table(key)
    out_path = "../data/training_table.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())