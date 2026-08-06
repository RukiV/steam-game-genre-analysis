import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.utils import GAMES, GAME_IDS, GENRE_IDS, GAME_GENRES


def prepare_regression_features(df):
    model_df = df.copy()
    model_df = model_df.dropna(subset=['playtime_forever', 'review_length', 'word_count',
                                        'vader_compound', 'num_games_owned', 'num_reviews',
                                        'votes_up'])

    features = pd.DataFrame()
    features['playtime_forever'] = np.log1p(model_df['playtime_forever'])
    features['review_length'] = np.log1p(model_df['review_length'])
    features['word_count'] = np.log1p(model_df['word_count'])
    features['num_games_owned'] = np.log1p(model_df['num_games_owned'])
    features['num_reviews'] = np.log1p(model_df['num_reviews'])
    features['votes_up'] = np.log1p(model_df['votes_up'])
    features['steam_purchase'] = model_df['steam_purchase'].astype(int)
    features['early_access'] = model_df['written_during_early_access'].astype(int)

    for app_id in GAME_IDS:
        name = GAMES[app_id].replace(' ', '_').replace("'", '').replace(':', '')
        features[name] = (model_df['app_id'] == app_id).astype(int)

    genres_for_app = model_df['app_id'].map(GAME_GENRES)
    for genre in GENRE_IDS:
        features[f'genre_{genre}'] = genres_for_app.apply(
            lambda tags: 1 if isinstance(tags, list) and genre in tags else 0
        )

    return features, model_df


def linear_regression_model(df, target_col='vader_compound'):
    features, model_df = prepare_regression_features(df)
    target = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    results = {
        'model': model,
        'scaler': scaler,
        'r2_score': round(r2_score(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'feature_importance': dict(zip(features.columns,
                                        [round(c, 4) for c in model.coef_])),
        'intercept': round(model.intercept_, 4),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    return results


def logistic_regression_voted_up(df):
    features, model_df = prepare_regression_features(df)
    features['vader_compound'] = model_df['vader_compound'].values
    target = model_df['voted_up'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results = {
        'model': model,
        'scaler': scaler,
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'auc_roc': round(roc_auc_score(y_test, y_prob), 4),
        'feature_importance': dict(zip(features.columns,
                                        [round(c, 4) for c in model.coef_[0]])),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
    return results
