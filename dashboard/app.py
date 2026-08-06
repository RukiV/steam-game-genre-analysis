import sys
import os
import gc
import faulthandler
faulthandler.enable()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

pd.options.mode.string_storage = 'python'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils import PROCESSED_DIR, GENRES, GENRE_IDS
from src.eda import basic_statistics, per_game_statistics, per_genre_statistics
from src.nlp_analysis import apply_vader, get_tfidf_features, top_tfidf_terms, get_tfidf_by_genre
from src.network import genre_relationship_network, genre_commentary_network, genre_word_network, draw_network
from src.regression import linear_regression_model, logistic_regression_voted_up

plt.close('all')
gc.collect()

st.set_page_config(
    page_title='Steam Game Analytics Dashboard',
    page_icon=':video_game:',
    layout='wide',
    initial_sidebar_state='expanded',
)


@st.cache_resource
def load_data():
    reviews_path = f'{PROCESSED_DIR}/reviews_clean.csv'
    details_path = f'{PROCESSED_DIR}/../raw/app_details.csv'
    players_path = f'{PROCESSED_DIR}/../raw/player_counts.csv'

    df_reviews = None
    df_details = None
    df_players = None

    if os.path.exists(reviews_path):
        df_reviews = pd.read_csv(reviews_path, parse_dates=['review_date'])
        df_reviews['app_id'] = df_reviews['app_id'].astype(int)
        if 'vader_compound' not in df_reviews.columns:
            sample_for_vader = df_reviews.sample(n=min(50000, len(df_reviews)), random_state=42)
            df_vader_partial = apply_vader(sample_for_vader)
            df_reviews['vader_compound'] = df_vader_partial['vader_compound']
            del sample_for_vader, df_vader_partial
            gc.collect()
        df_reviews['vader_sentiment_label'] = pd.cut(
            df_reviews['vader_compound'],
            bins=[-1, -0.05, 0.05, 1],
            labels=['negative', 'neutral', 'positive']
        )
        for col in ['review_text', 'review_text_clean']:
            if col in df_reviews.columns:
                del df_reviews[col]
        gc.collect()

    if os.path.exists(details_path):
        df_details = pd.read_csv(details_path)

    if os.path.exists(players_path):
        df_players = pd.read_csv(players_path)

    gc.collect()
    return df_reviews, df_details, df_players


@st.cache_resource
def load_review_texts():
    reviews_path = f'{PROCESSED_DIR}/reviews_clean.csv'
    if not os.path.exists(reviews_path):
        return None
    texts = pd.read_csv(reviews_path, usecols=['review_id', 'review_text_clean'])
    return texts


df, df_details, df_players = load_data()
gc.collect()

st.sidebar.markdown('# :video_game: Steam Dashboard')
st.sidebar.markdown('---')

if df is not None:
    selected_genres = st.sidebar.multiselect(
        'Filtreer genres',
        options=sorted([g.replace('_', ' ') for g in GENRE_IDS]),
        default=[],
    )
    if selected_genres:
        genre_app_ids = set()
        for sg in selected_genres:
            genre_key = sg.replace(' ', '_')
            genre_app_ids.update(GENRES.get(genre_key, []))
        df = df[df['app_id'].isin(genre_app_ids)]

    selected_games = st.sidebar.multiselect(
        'Filtreer speletjies',
        options=sorted(df['game_name'].unique()),
        default=sorted(df['game_name'].unique()),
    )
    filtered = df[df['game_name'].isin(selected_games)] if selected_games else df

    if not filtered.empty:
        try:
            min_date = filtered['review_date'].min()
            max_date = filtered['review_date'].max()
            has_dates = pd.notna(min_date) and pd.notna(max_date)
        except Exception:
            has_dates = False
    else:
        has_dates = False

    if has_dates:
        date_range = st.sidebar.date_input(
            'Datumbereik',
            value=(min_date.date(), max_date.date()),
            min_value=min_date.date(),
            max_value=max_date.date(),
        )
        if len(date_range) == 2:
            filtered = filtered[
                (filtered['review_date'].dt.date >= date_range[0]) &
                (filtered['review_date'].dt.date <= date_range[1])
            ]
    else:
        st.sidebar.warning('Geen datums beskikbaar vir geselekteerde filter nie')

    min_reviews = st.sidebar.slider('Minimum resensies', 0, 5000, 0, 100)
    if min_reviews > 0 and not filtered.empty:
        game_counts = filtered['game_name'].value_counts()
        valid_games = game_counts[game_counts >= min_reviews].index
        filtered = filtered[filtered['game_name'].isin(valid_games)]

else:
    filtered = None

st.sidebar.markdown('---')
st.sidebar.markdown('**Steam Game Analytics** | Data Analise Projek')

st.markdown('# :video_game: Steam Game Reviews & Popularity Analysis')
st.markdown('Interaktiewe dashboard vir die analise van Steam-speletjie-resensies, sentiment, genre-verwantskappe en gewildheid.')

if filtered is None or filtered.empty:
    st.warning('Geen data gelaai nie. Hardloer eers die notebook om data in te samel en skoon te maak.')
    st.info('```bash\ncd notebooks\njupyter notebook project.ipynb\n```')
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    ':bar_chart: Oorsig',
    ':speech_balloon: NLP & Netwerke',
    ':left_right_arrow: Vergelyk',
    ':crystal_ball: Voorspellings',
])

with tab1:
    try:
        st.header('Algemene Oorsig')

        stats = basic_statistics(filtered)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric('Totale Resensies', f"{stats['total_reviews']:,}",
                      help=f"Aantal resensies in die huidige filter. Hoe meer resensies, hoe betroubaarder die gemiddeldes. Bron: Steam Web API.")
        with col2:
            st.metric('Positief %', f"{stats['positive_pct']}%",
                      help=f"Persentasie resensies wat die speletjie aanbeveel (voted_up = Ja). Berekening: aanbeveel / totaal × 100. Bron: Steam Web API.")
        with col3:
            st.metric('Speletjies', stats['num_games'],
                      help=f"Aantal unieke speletjies in die huidige filter. Bron: Steam Web API.")
        with col4:
            st.metric('Tale', stats['num_languages'],
                      help=f"Aantal verskillende tale waarin resensies geskryf is. Bron: Steam Web API.")

        st.markdown('### Per Speletjie Oorsig')
        per_game = per_game_statistics(filtered)
        if not per_game.empty:
            st.dataframe(per_game.style.format({
                'total_reviews': '{:,.0f}',
                'positive': '{:,.0f}',
                'negative': '{:,.0f}',
                'positive_pct': '{:.1f}%',
                'avg_playtime': '{:.0f}',
                'steam_purchase_pct': '{:.1f}%',
            }), width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                per_game.sort_values('total_reviews').plot(
                    x='game', y='total_reviews', kind='barh', ax=ax,
                    color='#3498db', legend=False, width=0.8
                )
                ax.set_title('Totale Resensies per Speletjie', fontsize=14, fontweight='bold')
                ax.set_xlabel('Aantal Resensies')
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Totale resensies per speletjie. "
                    f"**Lees:** Hoe langer die staaf, hoe meer resensies. "
                    f"**Bron:** Steam Web API. "
                )

            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#2ecc71' if v > 70 else '#f39c12' if v > 50 else '#e74c3c'
                          for v in per_game.sort_values('positive_pct')['positive_pct']]
                per_game.sort_values('positive_pct').plot(
                    x='game', y='positive_pct', kind='barh', ax=ax,
                    color=colors, legend=False, width=0.8
                )
                ax.set_title('Positiewe Resensies %', fontsize=14, fontweight='bold')
                ax.set_xlabel('% Positief')
                ax.axvline(50, color='black', linestyle='--', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Persentasie resensies wat aanbeveel. "
                    f"**Lees:** Groen ≥ 70% (gunstig), oranje 50-69% (gemeng), rooi < 50% (ongunstig). "
                    f"**Bron:** Steam Web API. "
                    f"**Let wel:** Rou persentasie van die steekproef, nie Steam se geweegde telling nie. "
                )

        st.markdown('### Per Genre Oorsig')
        per_genre = per_genre_statistics(filtered)
        if not per_genre.empty:
            st.dataframe(per_genre.style.format({
                'total_reviews': '{:,.0f}',
                'positive': '{:,.0f}',
                'negative': '{:,.0f}',
                'positive_pct': '{:.1f}%',
                'avg_playtime': '{:.0f}',
                'avg_word_count': '{:.1f}',
                'num_games': '{:.0f}',
            }), width='stretch')

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(10, 5))
                per_genre.sort_values('total_reviews').plot(
                    x='genre', y='total_reviews', kind='barh', ax=ax,
                    color='#3498db', legend=False, width=0.8
                )
                ax.set_title('Totale Resensies per Genre', fontsize=14, fontweight='bold')
                ax.set_xlabel('Aantal Resensies')
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Totale resensies per genre. "
                    f"**Lees:** Genres oorvleuel — 'n speletjie kan in veelvuldige genres val. "
                    f"**Bron:** Steam Web API + genre-indeling. "
                )
            with col2:
                fig, ax = plt.subplots(figsize=(10, 5))
                colors_g = ['#2ecc71' if v > 70 else '#f39c12' if v > 50 else '#e74c3c'
                            for v in per_genre.sort_values('positive_pct')['positive_pct']]
                per_genre.sort_values('positive_pct').plot(
                    x='genre', y='positive_pct', kind='barh', ax=ax,
                    color=colors_g, legend=False, width=0.8
                )
                ax.set_title('Positiewe Resensies % per Genre', fontsize=14, fontweight='bold')
                ax.set_xlabel('% Positief')
                ax.axvline(50, color='black', linestyle='--', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Gemiddelde positiewe % per genre. "
                    f"**Lees:** Dieselfde kleurskaal: groen ≥ 70%, oranje 50-69%, rooi < 50%. "
                    f"**Bron:** Steam Web API. "
                    f"**Let wel:** 'n Genre met min speletjies/resensies is minder betroubaar. "
                )

        st.markdown('### Huidige Spelersgetalle')
        if df_players is not None and not df_players.empty:
            players_filtered = df_players[df_players['game_name'].isin(selected_games)]
            if not players_filtered.empty:
                col1, col2, col3, col4, col5 = st.columns(5)
                for i, (_, row) in enumerate(players_filtered.iterrows()):
                    cols = [col1, col2, col3, col4, col5]
                    with cols[i % 5]:
                        st.metric(
                            row['game_name'],
                            f"{row['player_count']:,}",
                            help=f"Huidige aantal spelers op Steam op die oomblik van laaste skraap. Bron: Steam Community API."
                        )
                st.caption(
                    f"**Wat:** Huidige gelyktydige spelers per speletjie (gewildheid). "
                    f"**Lees:** Hoe hoër die getal, hoe meer gewild is die speletjie nou. "
                    f"**Bron:** Steam Community API (laaste skraap). "
                )
            else:
                st.info('Geen spelersdata vir die geselekteerde speletjies nie.')
        else:
            st.info('Geen spelersdata beskikbaar nie (data/raw/player_counts.csv).')

        gc.collect()
    except Exception as e:
        st.warning(f'Oorsig tab kon nie laai nie: {e}')
        gc.collect()

with tab2:
    try:
        from wordcloud import WordCloud
        st.header('NLP Analise & Genre Netwerke')

        texts_df = load_review_texts()
        if texts_df is not None:
            filtered_en = filtered[filtered['language'] == 'english'].copy() if 'language' in filtered.columns else filtered.copy()
            filtered_en = filtered_en.merge(texts_df, on='review_id', how='left')
        else:
            filtered_en = filtered[filtered['language'] == 'english'].copy() if 'language' in filtered.columns else filtered.copy()
        gc.collect()

        if 'vader_compound' in filtered_en.columns:
            st.subheader('VADER Sentiment Verspreiding')
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(filtered_en['vader_compound'], bins=50, color='#1abc9c', edgecolor='black')
                ax.set_title('Compound Score')
                ax.set_xlabel('VADER Compound')
                ax.set_ylabel('Frekwensie')
                ax.axvline(-0.05, color='red', linestyle='--', alpha=0.5)
                ax.axvline(0.05, color='green', linestyle='--', alpha=0.5)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Verspreiding van VADER compound tellings (-1 = negatief, +1 = positief). "
                    f"**Lees:** Rooi stippellyn = -0.05 (negatief drempel), groen = +0.05 (positief drempel). "
                    f"**Bron:** VADER-sentiment analise op Engelse resensieteks. "
                    f"**Let wel:** Slegs Engelse resensies. "
                )
            with col2:
                sentiments = filtered_en.groupby('game_name')['vader_compound'].mean().sort_values()
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(sentiments.index, sentiments.values,
                        color=['#e74c3c' if v < 0 else '#2ecc71' for v in sentiments.values])
                ax.set_title('Gemiddelde VADER per Speletjie')
                ax.set_xlabel('Gem. Compound Score')
                ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Gemiddelde VADER compound telling per speletjie. "
                    f"**Lees:** Rooi = gemiddeld negatief, groen = gemiddeld positief. "
                    f"**Bron:** VADER-sentiment analise op Engelse resensieteks. "
                )

        st.subheader('Woordwolk')
        unique_games = filtered_en['game_name'].unique()
        if len(unique_games) > 0:
            game_for_wc = st.selectbox('Kies speletjie vir woordwolk:', ['Alle speletjies'] + sorted(unique_games))
            if game_for_wc == 'Alle speletjies':
                texts = filtered_en['review_text_clean'].dropna()
            else:
                texts = filtered_en[filtered_en['game_name'] == game_for_wc]['review_text_clean'].dropna()

            MAX_WC_SAMPLES = 500

            if len(texts) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('**Positiewe Resensies**')
                    pos_texts = filtered_en[filtered_en['voted_up']]['review_text_clean'].dropna()
                    if game_for_wc != 'Alle speletjies':
                        pos_texts = filtered_en[(filtered_en['game_name'] == game_for_wc) & filtered_en['voted_up']]['review_text_clean'].dropna()
                    if len(pos_texts) > 0:
                        pos_sample = pos_texts.sample(n=min(MAX_WC_SAMPLES, len(pos_texts)), random_state=42)
                        wc = WordCloud(width=500, height=300, background_color='white',
                                       colormap='Greens', max_words=50).generate(' '.join(pos_sample))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            f"**Wat:** Woordwolk van positiewe resensies (aanbeveel). "
                            f"**Lees:** Grootter woorde = meer algemeen in positiewe resensies. "
                            f"**Bron:** Engelse resensieteks. "
                            f"**Let wel:** Steekproef van maks 500 resensies. "
                        )
                with col2:
                    st.markdown('**Negatiewe Resensies**')
                    neg_texts = filtered_en[~filtered_en['voted_up']]['review_text_clean'].dropna()
                    if game_for_wc != 'Alle speletjies':
                        neg_texts = filtered_en[(filtered_en['game_name'] == game_for_wc) & ~filtered_en['voted_up']]['review_text_clean'].dropna()
                    if len(neg_texts) > 0:
                        neg_sample = neg_texts.sample(n=min(MAX_WC_SAMPLES, len(neg_texts)), random_state=42)
                        wc = WordCloud(width=500, height=300, background_color='white',
                                       colormap='Reds', max_words=50).generate(' '.join(neg_sample))
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.imshow(wc, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            f"**Wat:** Woordwolk van negatiewe resensies (nie aanbeveel nie). "
                            f"**Lees:** Grootter woorde = meer algemeen in negatiewe resensies. "
                            f"**Bron:** Engelse resensieteks. "
                            f"**Let wel:** Steekproef van maks 500 resensies. "
                        )

        gc.collect()

        if len(unique_games) > 0:
            st.subheader('TF-IDF Top Terme')
            selected_game_tfidf = st.selectbox('Kies speletjie vir TF-IDF:',
                                                list(unique_games),
                                                key='tfidf_select')
            try:
                game_texts = filtered_en[filtered_en['game_name'] == selected_game_tfidf]['review_text_clean'].dropna()
                if len(game_texts) > 30000:
                    game_texts = game_texts.sample(n=30000, random_state=42)
                tfidf_matrix, feature_names = get_tfidf_features(
                    game_texts.to_frame(name='review_text_clean'), ngram_range=(1, 1)
                )
                top_terms = top_tfidf_terms(tfidf_matrix, feature_names, 20)
                del tfidf_matrix, feature_names, game_texts
                gc.collect()
                if top_terms:
                    terms_df = pd.DataFrame(top_terms, columns=['Term', 'TF-IDF Score'])
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(terms_df, width='stretch')
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        terms_df_sorted = terms_df.sort_values('TF-IDF Score', ascending=True).tail(15)
                        ax.barh(terms_df_sorted['Term'], terms_df_sorted['TF-IDF Score'], color='#1abc9c')
                        ax.set_title(f'Top TF-IDF Terme: {selected_game_tfidf}')
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            f"**Wat:** Belangrikste woorde volgens TF-IDF vir {selected_game_tfidf}. "
                            f"**Lees:** Hoe hoër die telling, hoe meer uniek en belangrik is die woord vir hierdie speletjie. "
                            f"**Bron:** Enkelwoord-analise (unigrams) op Engelse resensies. "
                            f"**Let wel:** Maks 30 000 resensies per speletjie. "
                        )
            except Exception as e:
                st.warning(f'Kon nie TF-IDF laai vir {selected_game_tfidf} nie: {e}')

            st.subheader('TF-IDF per Genre')
            selected_genre_tfidf = st.selectbox('Kies genre vir TF-IDF:',
                                                 sorted([g.replace('_', ' ') for g in GENRE_IDS]),
                                                 key='genre_tfidf')
            try:
                genre_tfidf_key = selected_genre_tfidf.replace(' ', '_')
                genre_texts = filtered_en[filtered_en[f'genre_{genre_tfidf_key}'] == 1]['review_text_clean'].dropna()
                if len(genre_texts) > 30000:
                    genre_texts = genre_texts.sample(n=30000, random_state=42)
                genre_terms = get_tfidf_by_genre(
                    genre_texts.to_frame(name='review_text_clean').assign(**{f'genre_{genre_tfidf_key}': 1}),
                    f'genre_{genre_tfidf_key}', top_n=20
                )
                del genre_texts
                gc.collect()
                if genre_terms:
                    genre_terms_df = pd.DataFrame(genre_terms, columns=['Term', 'TF-IDF Score'])
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(genre_terms_df, width='stretch')
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        genre_terms_sorted = genre_terms_df.sort_values('TF-IDF Score', ascending=True).tail(15)
                        ax.barh(genre_terms_sorted['Term'], genre_terms_sorted['TF-IDF Score'], color='#9b59b6')
                        ax.set_title(f'Top TF-IDF Terme: {selected_genre_tfidf}')
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            f"**Wat:** Belangrikste woorde volgens TF-IDF vir die {selected_genre_tfidf}-genre. "
                            f"**Lees:** Hoe hoër die telling, hoe meer kenmerkend is die woord vir hierdie genre. "
                            f"**Bron:** Enkelwoord-analise (unigrams) op Engelse resensies. "
                        )
            except Exception as e:
                st.warning(f'Kon nie genre TF-IDF laai vir {selected_genre_tfidf} nie: {e}')

        gc.collect()

        st.subheader('Genre Netwerke')
        st.markdown('Drie netwerke wys hoe genres verwant is: gedeelde speletjies, '
                    'kommentaar-ooreenkoms (TF-IDF) en ko-voorkoms van woorde.')

        G_rel = genre_relationship_network()
        fig, ax = plt.subplots(figsize=(10, 7))
        draw_network(G_rel, ax=ax, title='Genre Verwantskap: Gedeelde Speletjies',
                     draw_edge_labels=True)
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            f"**Wat:** Nodus = genre (grootte = aantal speletjies), kantlyn = aantal "
            f"gedeelde speletjies tussen twee genres. "
            f"**Lees:** Dikker kantlyne = meer gedeelde speletjies; genres sonder kantlyn "
            f"deel geen speletjies nie. "
            f"**Bron:** Genre-indeling in src/utils.py. "
        )

        try:
            G_comm = genre_commentary_network(filtered_en)
            if G_comm.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                draw_network(G_comm, ax=ax, title='Genre Kommentaar: TF-IDF Vokabulêre',
                             draw_edge_labels=True)
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Nodus = genre (grootte = aantal Engelse resensies), "
                    f"kantlyn = cosinus-ooreenkoms tussen genre-vokabulêre (TF-IDF-sentroïede). "
                    f"**Lees:** Sterk kantlyne = genres gebruik dieselfde woorde in resensies. "
                    f"Slegs die sterkste bande (bo gemiddeld + 0.5·std) word gewys. "
                    f"**Bron:** Engelse resensieteks, enkelwoord-TF-IDF. "
                )
            else:
                st.info('Nie genoeg Engelse resensies vir die kommentaarnetwerk nie.')
        except Exception as e:
            st.warning(f'Kon nie kommentaarnetwerk laai nie: {e}')

        genre_for_wn = st.selectbox('Kies genre vir woordnetwerk:',
                                    sorted([g.replace('_', ' ') for g in GENRE_IDS]),
                                    key='genre_wordnet')
        genre_wn_key = genre_for_wn.replace(' ', '_')
        try:
            G_word = genre_word_network(filtered_en, genre_wn_key)
            if G_word.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                draw_network(G_word, ax=ax, title=f'Woordnetwerk: {genre_for_wn}',
                             node_color='#9b59b6')
                st.pyplot(fig)
                plt.close(fig)
                st.caption(
                    f"**Wat:** Nodus = top-woord (grootte = frekwensie), kantlyn = aantal "
                    f"resensies waarin beide woorde saam voorkom. "
                    f"**Lees:** Woorde wat dikwels saam gebruik word, vorm clusters — "
                    f"die genre se 'tipe kommentaar'. "
                    f"**Bron:** Engelse resensies, steekproef van maks 500. "
                )
            else:
                st.info('Nie genoeg Engelse resensies vir hierdie genre se woordnetwerk nie.')
        except Exception as e:
            st.warning(f'Kon nie woordnetwerk laai nie: {e}')

        gc.collect()
    except Exception as e:
        st.warning(f'NLP & Netwerke tab kon nie laai nie: {e}')
        gc.collect()
with tab3:
    try:
        st.header('Vergelyk')
        compare_mode = st.radio('Vergelyk modus:', ['Speletjies', 'Genres'], horizontal=True)

        if compare_mode == 'Genres':
            compare_genres = st.multiselect(
                'Kies genres om te vergelyk:',
                sorted([g.replace('_', ' ') for g in GENRE_IDS]),
                default=sorted([g.replace('_', ' ') for g in GENRE_IDS])[:3],
            )
            if compare_genres:
                genre_keys = [g.replace(' ', '_') for g in compare_genres]
                compare_data = filtered[filtered[[f'genre_{gk}' for gk in genre_keys]].sum(axis=1) > 0]

                st.subheader('Resensie Lengte vs Sentiment')
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for gk in genre_keys:
                        g = compare_data[compare_data[f'genre_{gk}'] == 1]
                        display_name = gk.replace('_', ' ')
                        if not g.empty:
                            sample = g.sample(min(5000, len(g))) if len(g) > 5000 else g
                            ax.scatter(sample['word_count'], sample['vader_compound'],
                                       label=display_name, alpha=0.3, s=5)
                    ax.set_xlabel('Woordtelling')
                    ax.set_ylabel('VADER Compound')
                    ax.set_title('Resensielengte vs Sentiment per Genre')
                    ax.legend()
                    ax.set_xlim(0, 200)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"**Wat:** Verwantskap tussen resensielengte en sentiment per genre. "
                        f"**Lees:** Punte bo 0 is positief, onder 0 is negatief. "
                        f"**Bron:** Engelse resensies, VADER-sentiment. "
                        f"**Let wel:** Steekproef van maks 5 000 per genre, x-as beperk tot 200 woorde. "
                    )
                except Exception as e:
                    st.warning(f'Kon nie grafiek laai nie: {e}')

                st.subheader('Speeltyd vs Aanbeveling per Genre')
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for gk in genre_keys:
                        g = compare_data[compare_data[f'genre_{gk}'] == 1]
                        display_name = gk.replace('_', ' ')
                        if not g.empty:
                            sample = g.sample(min(3000, len(g))) if len(g) > 3000 else g
                            sample = sample[sample['playtime_forever'] <= 2000]
                            ax.scatter(sample['playtime_forever'], sample['voted_up'].astype(int),
                                       label=display_name, alpha=0.2, s=3)
                    ax.set_xlabel('Speeltyd (ure)')
                    ax.set_ylabel('Aanbeveel (0=Nee, 1=Ja)')
                    ax.set_title('Speeltyd vs Aanbeveling per Genre')
                    ax.legend()
                    ax.set_xlim(0, 2000)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"**Wat:** Verwantskap tussen speeltyd en aanbeveling per genre. "
                        f"**Lees:** Punte bo 0.5 = aanbeveel, onder 0.5 = nie aanbeveel nie. "
                        f"**Bron:** Steam Web API. "
                        f"**Let wel:** Steekproef van maks 3 000 per genre, speeltyd ≤ 2 000 uur. "
                    )
                except Exception as e:
                    st.warning(f'Kon nie grafiek laai nie: {e}')
        else:
            compare_games = st.multiselect(
                'Kies speletjies om te vergelyk:',
                sorted(filtered['game_name'].unique()),
                default=sorted(filtered['game_name'].unique())[:3],
            )

            if compare_games:
                compare_data = filtered[filtered['game_name'].isin(compare_games)]

                st.subheader('Resensie Lengte vs Sentiment')
                try:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for game in compare_games:
                        g = compare_data[compare_data['game_name'] == game]
                        sample = g.sample(min(5000, len(g))) if len(g) > 5000 else g
                        ax.scatter(sample['word_count'], sample['vader_compound'],
                                   label=game, alpha=0.3, s=5)
                    ax.set_xlabel('Woordtelling')
                    ax.set_ylabel('VADER Compound')
                    ax.set_title('Resensielengte vs Sentiment')
                    ax.legend()
                    ax.set_xlim(0, 200)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"**Wat:** Verwantskap tussen resensielengte en sentiment per speletjie. "
                        f"**Lees:** Punte bo 0 is positief, onder 0 is negatief. "
                        f"**Bron:** Engelse resensies, VADER-sentiment. "
                        f"**Let wel:** Steekproef van maks 5 000 per speletjie, x-as beperk tot 200 woorde. "
                    )
                except Exception as e:
                    st.warning(f'Kon nie resensielengte grafiek laai nie: {e}')

                st.subheader('Speeltyd vs Aanbeveling')
                try:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    for game in compare_games:
                        g = compare_data[compare_data['game_name'] == game]
                        sample = g.sample(min(3000, len(g))) if len(g) > 3000 else g
                        sample = sample[sample['playtime_forever'] <= 2000]
                        ax.scatter(sample['playtime_forever'], sample['voted_up'].astype(int),
                                   label=game, alpha=0.2, s=3)
                    ax.set_xlabel('Speeltyd (ure)')
                    ax.set_ylabel('Aanbeveel (0=Nee, 1=Ja)')
                    ax.set_title('Speeltyd vs Aanbeveling')
                    ax.legend()
                    ax.set_xlim(0, 2000)
                    st.pyplot(fig)
                    plt.close(fig)
                    st.caption(
                        f"**Wat:** Verwantskap tussen speeltyd en aanbeveling per speletjie. "
                        f"**Lees:** Punte bo 0.5 = aanbeveel, onder 0.5 = nie aanbeveel nie. "
                        f"**Bron:** Steam Web API. "
                        f"**Let wel:** Steekproef van maks 3 000 per speletjie, speeltyd ≤ 2 000 uur. "
                    )
                except Exception as e:
                    st.warning(f'Kon nie speeltyd grafiek laai nie: {e}')

        gc.collect()
    except Exception as e:
        st.warning(f'Vergelyk tab kon nie laai nie: {e}')
        gc.collect()

with tab4:
    try:
        st.header('Voorspellings & Regressie')
        st.markdown('Lineêre regressie en logistiese regressie modelle gebaseer op resensie-kenmerke.')

        if not filtered.empty and 'vader_compound' in filtered.columns:
            model_data = filtered.dropna(subset=['playtime_forever', 'review_length',
                                                  'vader_compound', 'word_count'])
            if len(model_data) > 100:
                try:
                    with st.spinner('Bou lineêre regressie model...'):
                        lr = linear_regression_model(model_data)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Lineêre Regressie R²', lr['r2_score'],
                                  help=f"R² meet hoe goed die model die variansie in VADER compound verduidelik. 1 = perfek, 0 = geen verklaring. Bron: Lineêre regressie op {lr['n_train']} oefen- en {lr['n_test']} toetsresensies.")
                    with col2:
                        st.metric('RMSE', lr['rmse'],
                                  help=f"Gemiddelde kwadratiese fout — die gemiddelde verskil tussen voorspelde en werklike VADER compound. Laer is beter. Bron: Lineêre regressie.")

                    st.subheader('Top Kenmerke (Lineêre Regressie)')
                    feat_df = pd.DataFrame(
                        sorted(lr['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:10],
                        columns=['Kenmerk', 'Koëffisiënt']
                    )
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(feat_df, width='stretch')
                    with col2:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        colors_lr = ['#2ecc71' if v > 0 else '#e74c3c' for v in feat_df['Koëffisiënt']]
                        ax.barh(feat_df['Kenmerk'], feat_df['Koëffisiënt'], color=colors_lr)
                        ax.set_title('Kenmerkbelangrikheid')
                        ax.axvline(0, color='black', linestyle='-', alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig)
                        st.caption(
                            f"**Wat:** Kenmerkbelangrikheid vir lineêre regressie (VADER compound). "
                            f"**Lees:** Groen = positiewe invloed, rooi = negatiewe invloed. "
                            f"**Bron:** Lineêre regressie op {lr['n_train'] + lr['n_test']} resensies. "
                            f"**Let wel:** Kenmerke sluit speeltyd, woordtelling, genre en speletjie-duimmies in. "
                        )
                except Exception as e:
                    st.warning(f'Lineêre regressie kon nie voltooi nie: {e}')

                try:
                    with st.spinner('Bou logistiese regressie model...'):
                        logit = logistic_regression_voted_up(model_data)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric('Logistiese Regressie Akkuraatheid', f"{logit['accuracy']:.2%}",
                                  help=f"Persentasie korrekte voorspellings (voted_up = Ja/Nee). Bron: Logistiese regressie op {logit['n_train']} oefen- en {logit['n_test']} toetsresensies.")
                    with col2:
                        st.metric('AUC-ROC', f"{logit['auc_roc']:.3f}",
                                  help=f"AUC-ROC meet die model se vermoë om tussen positiewe en negatiewe resensies te onderskei. 1.0 = perfek, 0.5 = ewekansig. Bron: Logistiese regressie.")

                    st.subheader('Klassifikasie Verslag')
                    clf_report = pd.DataFrame(logit['classification_report']).transpose()
                    st.dataframe(clf_report.style.format({
                        'precision': '{:.2f}',
                        'recall': '{:.2f}',
                        'f1-score': '{:.2f}',
                        'support': '{:.0f}',
                    }), width='stretch')
                except Exception as e:
                    st.warning(f'Logistiese regressie kon nie voltooi nie: {e}')
            else:
                st.warning('Nie genoeg data vir regressie nie. Hardloop die skraper om meer data in te samel.')
        else:
            st.warning('Geen data beskikbaar nie. Hardloer eers die Jupyter notebook om data in te samel.')
    except Exception as e:
        st.warning(f'Voorspellings tab kon nie laai nie: {e}')
