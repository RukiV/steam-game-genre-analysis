import re
import time
import pandas as pd
import requests
from src.utils import GAMES, GAME_IDS, RAW_DIR, ensure_dirs

STEAMCHARTS_URL = 'https://steamcharts.com/app/{app_id}'


def scrape_historical_players(app_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(STEAMCHARTS_URL.format(app_id=app_id), timeout=30)
            if resp.status_code != 200:
                time.sleep(2)
                continue

            html = resp.text
            tbody = re.search(r'<tbody>(.*?)</tbody>', html, re.DOTALL)
            if not tbody:
                return []

            content = tbody.group(1)
            rows = re.findall(
                r'<td[^>]*class="month-cell[^"]*"[^>]*>\s*(.*?)\s*</td>\s*'
                r'<td[^>]*class="right num-f[^"]*"[^>]*>\s*(.*?)\s*</td>\s*'
                r'<td[^>]*class="right num-p[^"]*"[^>]*>\s*(.*?)\s*</td>\s*'
                r'<td[^>]*class="right\s*gainorloss"[^>]*>\s*(.*?)\s*</td>\s*'
                r'<td[^>]*class="right num"[^>]*>\s*(.*?)\s*</td>',
                html, re.DOTALL
            )

            results = []
            for month_label, avg, change, pct, peak in rows:
                label = month_label.strip()
                if label == 'Last 30 Days':
                    continue
                try:
                    avg_clean = avg.strip().replace(',', '')
                    peak_clean = peak.strip().replace(',', '')
                    results.append({
                        'app_id': app_id,
                        'game_name': GAMES.get(app_id, ''),
                        'month': label,
                        'avg_players': float(avg_clean),
                        'peak_players': float(peak_clean),
                    })
                except ValueError:
                    continue
            return results

        except requests.RequestException:
            time.sleep(2)

    return []


def scrape_all_games():
    ensure_dirs()
    all_history = []
    for app_id in GAME_IDS:
        name = GAMES[app_id]
        print(f'SteamCharts: {name}...', end=' ', flush=True)
        data = scrape_historical_players(app_id)
        all_history.extend(data)
        print(f'{len(data)} months')
        time.sleep(1)

    df = pd.DataFrame(all_history)
    out_path = f'{RAW_DIR}/player_history.csv'
    df.to_csv(out_path, index=False)
    print(f'\nSaved {len(all_history)} rows to player_history.csv')
    return df


def prepare_monthly_players(df_history):
    if df_history is None or df_history.empty:
        return pd.DataFrame()

    df = df_history.copy()
    df['date'] = pd.to_datetime('01 ' + df['month'], format='%d %B %Y')
    df = df.sort_values(['app_id', 'date'])
    return df


if __name__ == '__main__':
    scrape_all_games()
