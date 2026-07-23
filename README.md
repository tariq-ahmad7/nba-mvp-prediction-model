# nba-mvp-prediction-model

## Overview 

I built a project that uses machine learning to predict NBA MVP candidates based on historical player statistics. The model analyzes regular-season statistics from 2000-2026 to estimate the probability of a player having an MVP-caliber season. 

## Technologies Used 
- Python
- pandas
- scikit-learn
- Kaggle API
- Jupyter Notebook

## Features: 

- Points per game
- Rebounds per game 
- Assists per game 
- Steals per game 
- Blocks per game 
- Field goal percentage 
- Three-point percentage 
- Free throw percentage 
- Minutes per game 

## Approach

I built a classification model using Logistic Regression to predict the probability that a player's season resulted in an MVP award. To make predictions more accurate, the model uses:
- Feature engineering to organize player statistics into meaningful season-level data
- Class balancing to handle the rarity of MVP-winning seasons
- CalibratedClassifierCV to produce more reliable MVP probability estimates 

## Important Note 

The usage of CalibratedClassifierCV resulted in numbers that look relatively low for the probabilities. Something important to note is that every season can include around 450–550 players. The model uses this metric to calculate probabilities leading to an average player's chance of winning MVP being 0.2%. Therefore a prediction of 6% represents a very strong MVP candidate compared to the league average, while a prediction around 28% represents an exceptionally high probability in this context.

## Results

The model was evaluated by comparing its predictions against historical MVP winners since 2000, resulting in an exact match accuracy of 23.1% and real MVPs in the model's top 5 at a rate of 84.6%

