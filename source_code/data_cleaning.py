import pandas as pd
import os
import kagglehub

def load_and_clean_data():
    path = kagglehub.dataset_download("chevronronson/nba-stats-dataset")
    csv_folder = os.path.join(path, "csv")
    player_df = pd.read_csv(os.path.join(csv_folder, "player_boxscores.csv"))

    season_stats = player_df.groupby(['season_year', 'player_name', 'team_abbreviation']).agg(
        games_played=('game_id', 'count'),
        mpg=('min', 'mean'), ppg=('pts', 'mean'), rpg=('reb', 'mean'),
        apg=('ast', 'mean'), spg=('stl', 'mean'), bpg=('blk', 'mean'),
        fg_pct=('fg_pct', 'mean'), fg3_pct=('fg3_pct', 'mean'), ft_pct=('ft_pct', 'mean')
    ).reset_index()

    season_stats = season_stats.round({
        'mpg': 1, 'ppg': 1, 'rpg': 1, 'apg': 1, 'spg': 1, 'bpg': 1,
        'fg_pct': 3, 'fg3_pct': 3, 'ft_pct': 3
    })
    return season_stats[season_stats['season_year'] >= '2000-01']


def load_mvp_history(path="mvp_winners.csv"):
    return pd.read_csv(path, skiprows=1)