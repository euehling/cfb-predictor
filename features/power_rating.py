"""
Power rating engine — faithful Python re-implementation of Em's Excel
CFP-style power-rating workbook (CFP_Rankings_2025.xlsm).

Weights are intentionally FIXED to match the workbook exactly (see WEIGHTS
below). Do not auto-tune these in the primary pipeline — see
weekly_shadow_compare.py for the separate fixed-vs-learned experiment.

Formula per game (mirrors the 'Alabama' sheet / any team sheet in the .xlsm):

    win_pts       = WIN_POINTS if result == "Win" else 0
    road_bonus    = ROAD_BONUS if (locale == "Away" and result == "Win") else 0
                    NOTE: workbook only checks pts_for != "" (i.e. game played)
                    and locale == "Away", not result — replicated as-is below.
    mov_bonus     = MOV_BONUS if (pts_for - pts_against) > (MOV_THRESHOLD - 1) else 0
    rank_bonus    = 26 - opponent_ap_rank if opponent is AP top 25, else 0
    game_total    = win_pts + road_bonus + mov_bonus + rank_bonus
    loss_penalty  = LOSS_PENALTY if result == "Loss" else 0
    loss_mov_pen  = MOV_BONUS if (pts_against - pts_for) > (MOV_THRESHOLD - 1) else 0
    tier_mult     = TIER_1 / TIER_2 / TIER_3 depending on opponent tier
    neutral_mult  = 1.5 if locale == "Neutral" else 1.0

    adjusted      = (game_total * tier_mult * neutral_mult)
                    - (loss_penalty + loss_mov_pen) * opponent_tier

    sos_quality_pts = lookup(opponent_tier + locale) + SOS_OFFSET
        lookup table:
            1Away: 5.5   1Neutral: 4.5   1Home: 4.0
            2Away: 3.5   2Neutral: 2.5   2Home: 2.0
            3Away: 2.0   3Neutral: 1.0   3Home: 1.0

Team-season totals:
    power_rating   = sum(adjusted) across all games played
    sos_avg        = sum(sos_quality_pts) / games_played
"""

from dataclasses import dataclass, field
from enum import Enum


class Locale(str, Enum):
    HOME = "Home"
    AWAY = "Away"
    NEUTRAL = "Neutral"


# ---- Fixed weights, pulled verbatim from the 'All' sheet of the workbook ----
WEIGHTS = {
    "WIN_POINTS": 50,
    "MOV_BONUS": 10,
    "MOV_THRESHOLD": 10,
    "ROAD_BONUS": 8,
    "TIER_1": 2.0,
    "TIER_2": 1.6,
    "TIER_3": 1.0,
    "LOSS_PENALTY": 25,
    "SOS_OFFSET": 10,
}

# SOS quality-points lookup table, keyed "{tier}{locale}" exactly as the
# workbook's CONCATENATE(tier, locale) key does.
SOS_TABLE = {
    "1Away": 5.5, "1Neutral": 4.5, "1Home": 4.0,
    "2Away": 3.5, "2Neutral": 2.5, "2Home": 2.0,
    "3Away": 2.0, "3Neutral": 1.0, "3Home": 1.0,
}


def ap_rank_bonus(opponent_ap_rank: int | None) -> float:
    """Rank Bonus column: 26 - rank for AP top-25 opponents, else 0."""
    if opponent_ap_rank is None or opponent_ap_rank > 25 or opponent_ap_rank < 1:
        return 0.0
    return 26 - opponent_ap_rank


def tier_from_ap_rank(ap_rank: int | None) -> int:
    """
    Mirrors the 'Tiers & AP Rank' sheet's Tier column derivation:
    Tier 1: ranked/highly regarded FBS teams
    Tier 2: solid but unranked FBS teams
    Tier 3: weak FBS / FCS teams
    In the source workbook this was manually curated per team. For a
    programmatic pipeline, approximate with AP rank + a talent/SP+ cutoff
    (wired up once CFBD data is loaded) rather than hardcoding — see
    build_tiers() in features/cfbd_pull.py.
    """
    raise NotImplementedError(
        "Tier assignment is manually curated in the workbook. "
        "Programmatic version wired up in features/cfbd_pull.py::build_tiers()."
    )


@dataclass
class GameResult:
    opponent: str
    opponent_tier: int          # 1, 2, or 3
    opponent_ap_rank: int | None  # None if unranked
    locale: Locale
    pts_for: int
    pts_against: int


@dataclass
class GameScore:
    win_pts: float = 0.0
    road_bonus: float = 0.0
    mov_bonus: float = 0.0
    rank_bonus: float = 0.0
    game_total: float = 0.0
    loss_penalty: float = 0.0
    loss_mov_penalty: float = 0.0
    adjusted: float = 0.0
    sos_quality_pts: float = 0.0
    result: str = ""


def score_game(g: GameResult, weights: dict = WEIGHTS) -> GameScore:
    """Replicates one row of a team sheet (e.g. Alabama!A3:Q3)."""
    result = "Win" if g.pts_for > g.pts_against else "Loss"

    win_pts = weights["WIN_POINTS"] if result == "Win" else 0
    # Workbook's road bonus checks locale=="Away" only (not tied to result) —
    # replicated exactly, including that quirk.
    road_bonus = weights["ROAD_BONUS"] if g.locale == Locale.AWAY else 0
    mov = g.pts_for - g.pts_against
    mov_bonus = weights["MOV_BONUS"] if mov > (weights["MOV_THRESHOLD"] - 1) else 0
    rank_bonus = ap_rank_bonus(g.opponent_ap_rank)

    game_total = win_pts + road_bonus + mov_bonus + rank_bonus

    loss_mov = g.pts_against - g.pts_for
    loss_penalty = weights["LOSS_PENALTY"] if result == "Loss" else 0
    loss_mov_penalty = weights["MOV_BONUS"] if loss_mov > (weights["MOV_THRESHOLD"] - 1) else 0

    tier_mult = {1: weights["TIER_1"], 2: weights["TIER_2"], 3: weights["TIER_3"]}[g.opponent_tier]
    neutral_mult = 1.5 if g.locale == Locale.NEUTRAL else 1.0

    adjusted = (game_total * tier_mult * neutral_mult) - (loss_penalty + loss_mov_penalty) * g.opponent_tier

    sos_key = f"{g.opponent_tier}{g.locale.value}"
    sos_quality_pts = SOS_TABLE.get(sos_key, 0.0) + weights["SOS_OFFSET"]

    return GameScore(
        win_pts=win_pts, road_bonus=road_bonus, mov_bonus=mov_bonus,
        rank_bonus=rank_bonus, game_total=game_total,
        loss_penalty=loss_penalty, loss_mov_penalty=loss_mov_penalty,
        adjusted=adjusted, sos_quality_pts=sos_quality_pts, result=result,
    )


@dataclass
class TeamSeasonPower:
    team: str
    games: list = field(default_factory=list)  # list[GameScore]

    @property
    def power_rating(self) -> float:
        return sum(g.adjusted for g in self.games)

    @property
    def sos_avg(self) -> float:
        if not self.games:
            return 0.0
        return sum(g.sos_quality_pts for g in self.games) / len(self.games)

    @property
    def wins(self) -> int:
        return sum(1 for g in self.games if g.result == "Win")

    @property
    def losses(self) -> int:
        return sum(1 for g in self.games if g.result == "Loss")


def compute_weekly_power_ratings(team: str, games: list[GameResult]) -> list[TeamSeasonPower]:
    """
    Given a chronologically-ordered list of a team's games in a season,
    return the team's power-rating trajectory AFTER each week (i.e. a
    snapshot after 1 game, after 2 games, ... after N games) so we can
    join "power rating entering this matchup" onto the training table
    instead of leaking the full-season final rating into earlier weeks.
    """
    snapshots = []
    running = TeamSeasonPower(team=team)
    for g in games:
        running.games.append(score_game(g))
        # snapshot is a shallow copy of games list up to this point
        snapshots.append(TeamSeasonPower(team=team, games=list(running.games)))
    return snapshots


if __name__ == "__main__":
    # Sanity check against the workbook's own Alabama row 3 (vs Florida St, Away, 17-31)
    # Florida St in the 2025 workbook is Tier 2 per the Alabama sheet's VLOOKUP.
    # Verified against cached cell values: adjusted=-57.2, sos_quality_pts=13.5 ✓
    test_game = GameResult(
        opponent="Florida St", opponent_tier=2, opponent_ap_rank=None,
        locale=Locale.AWAY, pts_for=17, pts_against=31,
    )
    result = score_game(test_game)
    print("Alabama @ Florida St (17-31):", result)
