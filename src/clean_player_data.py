
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_player_ids(df, name):
    assert "nfl_id" in df.columns, f"{name}: missing nfl_id"
    assert df["nfl_id"].notna().all(), f"{name}: missing player IDs"
    assert df["nfl_id"].is_unique, f"{name}: duplicate player IDs"
    print(f"{name}: {len(df)} unique player IDs validated")


def clean_players(df):
    df = df.copy()

    validate_player_ids(df, "players")

    assert df["draft_year"].isin([2023, 2024, 2025]).all(), (
        "Unexpected draft year"
    )

    assert df["nfl_position"].notna().all(), (
        "Missing NFL positions"
    )

    df["birth_date"] = pd.to_datetime(
        df["birth_date"], errors="coerce"
    )

    draft_columns = [
        "draft_round",
        "draft_pick_within_round",
        "draft_overall_pick"
    ]

    for col in draft_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nDraft round distribution:")
    print(df["draft_round"].value_counts(dropna=False).sort_index())

    valid_round = df["draft_round"].between(1, 7)
    missing_round = df["draft_round"].isna()

    unexpected_round = ~(valid_round | missing_round)

    if unexpected_round.any():
        print("\nUnexpected draft round values:")
        print(
            df.loc[
                unexpected_round,
                ["nfl_id", "display_name", "draft_round"]
            ].to_string(index=False)
        )

    assert not unexpected_round.any(), (
        "Unexpected draft round values detected"
    )

    valid_pick = (
        df["draft_overall_pick"].between(1, 262)
        | df["draft_overall_pick"].isna()
    )

    assert valid_pick.all(), (
        "Invalid overall draft pick values"
    )

    valid_round_pick = (
        df["draft_pick_within_round"].between(1, 50)
        | df["draft_pick_within_round"].isna()
    )

    assert valid_round_pick.all(), (
        "Invalid pick within round"
    )

    draft_missing = df[draft_columns].isna()

    partial_draft_info = (
        draft_missing.any(axis=1)
        & ~draft_missing.all(axis=1)
    )

    if partial_draft_info.any():
        print("\nPlayers with incomplete draft information:")
        print(
            df.loc[
                partial_draft_info,
                ["nfl_id", "display_name"] + draft_columns
            ].to_string(index=False)
        )

    assert not partial_draft_info.any(), (
        "Inconsistent draft information"
    )

    df["is_drafted"] = df["draft_round"].notna()

    for col in draft_columns:
        df[col] = df[col].astype("Int64")

    print("\nPlayers by draft year:")
    print(df["draft_year"].value_counts().sort_index())

    print("\nPlayers by position:")
    print(df["nfl_position"].value_counts())

    print("\nMissing player metadata:")
    print(df.isna().sum())

    print("\nDrafted players:", df["is_drafted"].sum())
    print("Players without draft information:", (~df["is_drafted"]).sum())

    return df


def clean_career_success(df):
    df = df.copy()

    validate_player_ids(df, "career_success")

    career_columns = [
        "career_offensive_snaps",
        "career_defensive_snaps",
        "career_special_teams_snaps",
        "career_games_active",
        "career_games_started",
        "ap_all_pro_1st_team",
        "ap_all_pro_2nd_team",
        "pro_bowl_original_ballot"
    ]

    assert df[career_columns].notna().all().all(), (
        "Missing career statistics"
    )

    assert (df[career_columns] >= 0).all().all(), (
        "Negative career statistics"
    )

    assert (df[career_columns] % 1 == 0).all().all(), (
        "Non-integer career statistics"
    )

    assert (
        df["career_games_started"]
        <= df["career_games_active"]
    ).all(), "Games started exceeds games active"

    for col in career_columns:
        df[col] = df[col].astype("int64")

    print("\nPlayers with zero active games:",
          (df["career_games_active"] == 0).sum())

    print("\nCareer statistics summary:")
    print(df[career_columns].describe().T)

    return df


def main():
    players = pd.read_csv(RAW_DATA_DIR / "players.csv")

    career_success = pd.read_csv(
        RAW_DATA_DIR / "player_career_successes.csv"
    )

    print("Raw players shape:", players.shape)
    print("Raw career success shape:", career_success.shape)

    players_clean = clean_players(players)
    career_clean = clean_career_success(career_success)

    assert set(players_clean["nfl_id"]) == set(career_clean["nfl_id"]), (
        "Player IDs do not match across datasets"
    )

    assert len(players_clean) == len(players)
    assert len(career_clean) == len(career_success)

    print("\nCross-dataset player IDs validated")

    players_path = PROCESSED_DATA_DIR / "players_clean.csv"
    career_path = PROCESSED_DATA_DIR / "player_career_successes_clean.csv"

    players_clean.to_csv(players_path, index=False)
    career_clean.to_csv(career_path, index=False)

    print("\nCleaned datasets saved:")
    print(players_path)
    print(career_path)

    print("\nFinal shapes:")
    print("Players:", players_clean.shape)
    print("Career success:", career_clean.shape)


if __name__ == "__main__":
    main()
