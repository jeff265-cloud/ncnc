# FPL Mini-League Analytics (Streamlit)

Analyse a Classic mini-league from the official Fantasy Premier League (FPL) API.
**Default league:** `497796` (you can change it in the sidebar).

## Features
- **League Table & Results** — standings, GW points, total points
- **Manager Lineups** per Gameweek — starters/bench, captain/vice, and per-player GW points
- **Transfers** — per manager, filterable by GW
- **Captain Choices** — league-wide captain/vice distributions for a GW
- **Popular Picks** — most-picked players across the league for a GW
- **Differentials** — best low-ownership picks in the league for a GW (threshold slider)

## Run
```bash
pip install streamlit pandas matplotlib requests python-dateutil pytz
streamlit run app.py
```

## Notes
- FPL API endpoints are public for this data. The app caches responses with short TTLs.
- Times are shown in Australia/Brisbane by default.
- If your league is very large, loading all manager picks can take a little while.
