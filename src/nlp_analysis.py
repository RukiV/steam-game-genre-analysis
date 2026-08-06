import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def vader_sentiment(text):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        if not isinstance(text, str) or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
        return analyzer.polarity_scores(text)
    except ImportError:
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}


def apply_vader(df, batch_size=5000):
    results = []
    for i in range(0, len(df), batch_size):
        batch = df['review_text_clean'].iloc[i:i+batch_size]
        for text in batch:
            results.append(vader_sentiment(text))

    scores_df = pd.DataFrame(results)
    df = df.copy()
    df['vader_compound'] = scores_df['compound']
    df['vader_positive'] = scores_df['pos']
    df['vader_neutral'] = scores_df['neu']
    df['vader_negative'] = scores_df['neg']
    df['vader_sentiment_label'] = pd.cut(
        df['vader_compound'],
        bins=[-1, -0.05, 0.05, 1],
        labels=['negative', 'neutral', 'positive']
    )
    return df


def sentiment_agreement(df):
    df = df.copy()
    df['review_sentiment'] = df['voted_up']
    df['vader_sentiment_binary'] = df['vader_compound'] > 0.05
    df['agreement'] = df['review_sentiment'] == df['vader_sentiment_binary']
    agreement_rate = df['agreement'].mean()
    return agreement_rate, df


def get_tfidf_features(df, game_name=None, max_features=1000, ngram_range=(1, 1)):
    if game_name:
        texts = df[df['game_name'] == game_name]['review_text_clean']
    else:
        texts = df['review_text_clean']

    texts = texts.dropna()
    if len(texts) < 2:
        return None, None

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        min_df=5,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


def top_tfidf_terms(tfidf_matrix, feature_names, top_n=20):
    if tfidf_matrix is None:
        return []
    avg_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_scores.argsort()[::-1][:top_n]
    return [(feature_names[i], round(avg_scores[i], 4)) for i in top_indices]


def get_tfidf_per_game(df, top_n=20):
    all_top_terms = {}
    for app_id in df['app_id'].unique():
        name = df[df['app_id'] == app_id]['game_name'].iloc[0]
        tfidf_matrix, feature_names = get_tfidf_features(df, game_name=name)
        top_terms = top_tfidf_terms(tfidf_matrix, feature_names, top_n)
        if top_terms:
            all_top_terms[name] = top_terms
    return all_top_terms


def get_tfidf_by_genre(df, genre_col, top_n=20, max_features=1000):
    texts = df[df[genre_col] == 1]['review_text_clean'].dropna()
    if len(texts) < 2:
        return []
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1),
                                  stop_words='english', min_df=5)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return top_tfidf_terms(tfidf_matrix, feature_names, top_n)


def compare_genre_vocabulary(df, genre_a, genre_b, top_n=30):
    col_a = f'genre_{genre_a}'
    col_b = f'genre_{genre_b}'
    texts_a = df[df[col_a] == 1]['review_text_clean'].dropna()
    texts_b = df[df[col_b] == 1]['review_text_clean'].dropna()
    if len(texts_a) < 2 or len(texts_b) < 2:
        return None
    all_texts = pd.concat([texts_a, texts_b])
    labels = [1] * len(texts_a) + [0] * len(texts_b)
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 1),
                                  stop_words='english', min_df=5)
    tfidf = vectorizer.fit_transform(all_texts)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)
    model.fit(tfidf, labels)
    feat = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]
    top_idx = coefs.argsort()
    return {
        'top_in_a': [(feat[i], round(coefs[i], 4)) for i in top_idx[-10:][::-1]],
        'top_in_b': [(feat[i], round(coefs[i], 4)) for i in top_idx[:10]],
    }
