from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

def label_mvp_winners(season_stats, mvp_df):
    mvp_winners = set(zip(mvp_df['Season'], mvp_df['Player']))
    season_stats['won_mvp'] = season_stats.apply(
        lambda row: 1 if (row['season_year'], row['player_name']) in mvp_winners else 0,
        axis=1
    )
    return season_stats


def train_mvp_model(season_stats, features):
    X = season_stats[features]
    y = season_stats['won_mvp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_train, y_train)
    
    return calibrated_model

def add_top5_rank(season_stats, mvp_df):
    def real_winner_rank(season):
        season_data = season_stats[season_stats['season_year'] == season].sort_values(
            'mvp_probability', ascending=False
        ).reset_index(drop=True)

        real_winner = mvp_df[mvp_df['Season'] == season]['Player'].values[0]
        match = season_data[season_data['player_name'] == real_winner]
        if len(match) == 0:
            return None
        return match.index[0] + 1

    mvp_df = mvp_df.copy()
    mvp_df['model_rank_of_real_winner'] = mvp_df['Season'].apply(real_winner_rank)
    top5_accuracy = (mvp_df['model_rank_of_real_winner'] <= 5).mean()
    return top5_accuracy


def add_predictions(season_stats, model, features):
    season_stats['mvp_probability'] = model.predict_proba(season_stats[features])[:, 1]
    return season_stats


def evaluate_model(season_stats, mvp_df):
    model_picks = season_stats.loc[season_stats.groupby('season_year')['mvp_probability'].idxmax()]
    model_picks = model_picks[['season_year', 'player_name', 'mvp_probability']]

    comparison = model_picks.merge(mvp_df[['Season', 'Player']], left_on='season_year', right_on='Season')
    comparison['correct'] = comparison['player_name'] == comparison['Player']
    accuracy = comparison['correct'].mean()

    return comparison, accuracy