from source_code.data_cleaning import load_and_clean_data, load_mvp_history
from source_code.model_training import label_mvp_winners, train_mvp_model, add_predictions, evaluate_model, add_top5_rank


FEATURES = ['ppg', 'rpg', 'apg', 'spg', 'bpg', 'fg_pct', 'fg3_pct', 'ft_pct', 'mpg']

season_stats = load_and_clean_data()
mvp_df = load_mvp_history()
season_stats = label_mvp_winners(season_stats, mvp_df)

model = train_mvp_model(season_stats, FEATURES)
season_stats = add_predictions(season_stats, model, FEATURES)

comparison, accuracy = evaluate_model(season_stats, mvp_df)
top5_accuracy = add_top5_rank(season_stats, mvp_df)

def print_formatted_comparison(comparison,accuracy,top5_accuracy):
    print(f"\n{'Season':<10} {'Model Pick':<15} {'Probability':<10}   | {'Actual Winner':<22} {'Correct'}")
    print("-" * 75)
    for _, row in comparison.iterrows():
        prob_str = f"{row['mvp_probability']:.1%}"
        print(f"{row['season_year']:<10} {row['player_name']:<20} {prob_str:<8} | {row['Player']:<22} {row['correct']}")
    print("-" * 75)
    print(f"Exact match accuracy: {accuracy:.1%}")
    print(f"Real MVPs in the model's top 5: {top5_accuracy:.1%}")
    print(" ")

print_formatted_comparison(comparison, accuracy, top5_accuracy)