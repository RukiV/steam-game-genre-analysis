"""Genre-netwerkgrafieke: verwantskap, kommentaar-ooreenkoms en woordnetwerke.

Drie tipes netwerke:
- genre_relationship_network(): nodus = genre, kantlyn = gedeelde speletjies
- genre_commentary_network(): nodus = genre, kantlyn = TF-IDF-vokabulêre-ooreenkoms
- genre_word_network(): nodus = top-woord, kantlyn = ko-voorkoms in resensies
"""
import re
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import GENRES, GENRE_IDS


def genre_relationship_network():
    """Nodus = genre, kantlyn = aantal gedeelde speletjies tussen twee genres.

    Wys watter genres 'n familie vorm (bv. Shooter–Hero_Shooter deel
    Overwatch 2, TF2 en Marvel Rivals).
    """
    G = nx.Graph()
    for genre in GENRE_IDS:
        G.add_node(genre, label=genre.replace('_', ' '), size=len(GENRES[genre]))

    for i, g1 in enumerate(GENRE_IDS):
        for g2 in GENRE_IDS[i + 1:]:
            shared = len(set(GENRES[g1]) & set(GENRES[g2]))
            if shared > 0:
                G.add_edge(g1, g2, weight=shared)
    return G


def _english_filter(df, language):
    """Filter na 'n taal (bv. 'english') as die `language`-kolom bestaan."""
    if language and 'language' in df.columns:
        return df[df['language'] == language]
    return df


def _genre_texts(df, genre, sample_size):
    """Skoon resensietekste vir 'n genre, gesampel tot sample_size (WSL-veilig)."""
    col = f'genre_{genre}'
    texts = df[df[col] == 1]['review_text_clean'].dropna()
    n_total = len(texts)
    if n_total > sample_size:
        texts = texts.sample(sample_size, random_state=42)
    return texts, n_total


def _auto_threshold(sim):
    """Relatiewe drempel: gemiddelde + 0.5·std — hou net die sterkste kantlyne."""
    n = sim.shape[0]
    upper = [sim[i][j] for i in range(n) for j in range(i + 1, n)]
    if not upper:
        return 1.0
    return float(np.mean(upper) + 0.5 * np.std(upper))


def genre_commentary_network(df, max_features=1000, sample_size=3000,
                             min_df=5, threshold='auto', language='english'):
    """Nodus = genre, kantlyn = cosinus-ooreenkoms tussen genre-vokabulêre.

    TF-IDF word op al die genre-tekste saam gepas (globale IDF), dan word elke
    genre se gemiddelde TF-IDF-vektor (sentroid) bereken. Kantlyngewig = cosinus-
    ooreenkoms tussen sentroïede — hoe meer dieselfde woorde twee genres gebruik,
    hoe sterker die kantlyn. Kantlyne onder `threshold` word weggelaat.

    `threshold='auto'` (verstek) hou net kantlyne bo gemiddeld + 0.5·std —
    nodig omdat alle genre-vokabulêre baie oorvleuel (tipies >0.85).
    """
    df = _english_filter(df, language)

    genre_texts = {}
    genre_counts = {}
    for genre in GENRE_IDS:
        texts, n_total = _genre_texts(df, genre, sample_size)
        if len(texts) >= 2:
            genre_texts[genre] = texts
            genre_counts[genre] = n_total

    if len(genre_texts) < 2:
        return nx.Graph()

    all_texts = pd.concat(genre_texts.values())
    labels = []
    for genre, texts in genre_texts.items():
        labels.extend([genre] * len(texts))

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 1),
                                 stop_words='english', min_df=min_df)
    tfidf = vectorizer.fit_transform(all_texts)

    # Gemiddelde (sentroid) vektor per genre — hou dit spaars om geheue te spaar
    order = []
    centroids = []
    for genre, texts in genre_texts.items():
        idx = [j for j, lab in enumerate(labels) if lab == genre]
        sub = tfidf[idx]
        centroids.append(np.asarray(sub.mean(axis=0)).flatten())
        order.append(genre)

    sim = cosine_similarity(centroids)

    if threshold == 'auto':
        threshold = _auto_threshold(sim)

    G = nx.Graph()
    for i, genre in enumerate(order):
        G.add_node(genre, label=genre.replace('_', ' '), size=genre_counts[genre])

    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            w = float(sim[i][j])
            if w >= threshold:
                G.add_edge(order[i], order[j], weight=round(w, 3))
    return G


def genre_word_network(df, genre, top_n=20, sample_size=500, min_cooccur=2,
                       language='english'):
    """Nodus = top-woord in 'n genre, kantlyn = resensies waarin beide voorkom.

    Toon die 'tipe kommentaar' wat jy in 'n genre se resensies sien: woorde wat
    gereeld saam gebruik word, vorm clusters. Filter na Engels (verstek) sodat
    nie-Engelse woorde (bv. 'que', 'jogo') die netwerk nie verpolitie nie.
    """
    df = _english_filter(df, language)

    col = f'genre_{genre}'
    texts = df[df[col] == 1]['review_text_clean'].dropna()
    if len(texts) > sample_size:
        texts = texts.sample(sample_size, random_state=42)

    tokenized = []
    freq = {}
    for text in texts:
        words = [w for w in re.findall(r"[a-z']+", str(text).lower())
                 if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        tokenized.append(set(words))
        for w in words:
            freq[w] = freq.get(w, 0) + 1

    top_words = [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]
    if not top_words:
        return nx.Graph()
    top_set = set(top_words)

    G = nx.Graph()
    for w in top_words:
        G.add_node(w, label=w, size=freq[w])

    cooccur = {w1: {w2: 0 for w2 in top_words} for w1 in top_words}
    for words in tokenized:
        present = words & top_set
        for w1 in present:
            for w2 in present:
                if w1 < w2:
                    cooccur[w1][w2] += 1

    for w1 in top_words:
        for w2 in top_words:
            if w1 < w2 and cooccur[w1][w2] >= min_cooccur:
                G.add_edge(w1, w2, weight=cooccur[w1][w2])
    return G


def draw_network(G, ax=None, title='', node_size_max=2500, edge_width_max=6.0,
                 draw_edge_labels=False, node_color='#e67e22', edge_color='#2c3e50'):
    """Teken 'n netwerkgrafiek met spring-uitleg; nodusgrootte = `size`, kantlyn = `weight`."""
    import matplotlib.pyplot as plt

    if G.number_of_nodes() == 0:
        if ax is not None:
            ax.set_title(title)
            ax.axis('off')
        return ax

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, seed=42, weight='weight')

    sizes = [G.nodes[n].get('size', 1) for n in G.nodes]
    max_size = max(sizes) if sizes else 1
    node_sizes = [80 + (s / max_size) * (node_size_max - 80) for s in sizes]

    weights = [G.edges[e].get('weight', 1) for e in G.edges]
    max_w = max(weights) if weights else 1
    edge_widths = [0.5 + (w / max_w) * (edge_width_max - 0.5) for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color=node_color, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths,
                           alpha=0.55, edge_color=edge_color)
    labels = {n: G.nodes[n].get('label', n) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10)

    if draw_edge_labels and G.number_of_edges() <= 40:
        edge_labels = {(u, v): G.edges[u, v].get('weight', '')
                       for u, v in G.edges}
        nx.draw_networkx_edge_labels(G, pos, ax=ax, edge_labels=edge_labels,
                                     font_size=8)

    ax.set_title(title, fontweight='bold')
    ax.axis('off')
    return ax
