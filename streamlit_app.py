import os
import secrets
import urllib.parse
from datetime import datetime, timezone, timedelta

import requests
import streamlit as st

# === 1) СЕКРЕТИ / ОКОЛНА СРЕДА ==============================================
# Увери се, че имаме SUPABASE_DB_URL като env, защото db.py го чете от os.getenv
if "SUPABASE_DB_URL" in st.secrets:
    os.environ.setdefault("SUPABASE_DB_URL", st.secrets["SUPABASE_DB_URL"])

STRAVA_CLIENT_ID = st.secrets.get("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = st.secrets.get("STRAVA_CLIENT_SECRET")
STRAVA_REDIRECT_URI = st.secrets.get("STRAVA_REDIRECT_URI")  # напр. https://onflows.streamlit.app
if not (STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET and STRAVA_REDIRECT_URI):
    st.error("Липсват STRAVA_CLIENT_ID / STRAVA_CLIENT_SECRET / STRAVA_REDIRECT_URI в Secrets.")
    st.stop()

# === 2) ИМПОРТ НА БАЗАТА (след като сме сетнали SUPABASE_DB_URL) ============
from db import init_db, upsert_activities, upsert_activity_streams  # noqa: E402

# Автосъздаване на липсващи таблици (не трие нищо)
try:
    init_db(create_missing=True)
except Exception as e:
    st.error(f"Грешка при init_db(): {e}")

# === 3) STRAVA OAUTH + API ===================================================
AUTH_BASE = "https://www.strava.com/oauth/authorize"
TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"


def oauth_login_url() -> str:
    state = secrets.token_urlsafe(16)
    st.session_state["oauth_state"] = state
    params = {
        "client_id": STRAVA_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": STRAVA_REDIRECT_URI,
        "approval_prompt": "auto",
        "scope": "read,activity:read_all",
        "state": state,
    }
    return f"{AUTH_BASE}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(code: str) -> dict:
    data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    tok = r.json()
    return {
        "access_token": tok["access_token"],
        "refresh_token": tok["refresh_token"],
        "expires_at": datetime.fromtimestamp(tok["expires_at"], tz=timezone.utc),
        "athlete": tok.get("athlete", {}),
    }


def refresh_access_token(refresh_token: str) -> dict:
    data = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    r.raise_for_status()
    tok = r.json()
    return {
        "access_token": tok["access_token"],
        "refresh_token": tok["refresh_token"],
        "expires_at": datetime.fromtimestamp(tok["expires_at"], tz=timezone.utc),
    }


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def get_athlete_activities(access_token: str, after_ts: int | None = None, per_page: int = 100):
    params = {"per_page": per_page, "page": 1}
    if after_ts:
        params["after"] = after_ts
    out = []
    while True:
        r = requests.get(f"{API_BASE}/athlete/activities", headers=_auth_headers(access_token),
                         params=params, timeout=45)
        r.raise_for_status()
        batch = r.json()
        out.extend(batch)
        if len(batch) < per_page:
            break
        params["page"] += 1
    return out


def get_activity_streams(access_token: str, activity_id: int) -> dict:
    # high resolution ~ 1 Hz, key_by_type=true => лесно мапване
    params = {
        "keys": "time,latlng,altitude,distance,velocity_smooth,heartrate,cadence,watts,temperature",
        "key_by_type": "true",
        "resolution": "high",
    }
    r = requests.get(f"{API_BASE}/activities/{activity_id}/streams",
                     headers=_auth_headers(access_token), params=params, timeout=60)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


# === 4) UI ===================================================================
st.set_page_config(page_title="onFlows – Strava → Supabase", page_icon="🏃")
st.title("onFlows – Strava → Supabase")

# Обработка на OAuth callback (?code=...&state=...)
params = st.query_params
if "code" in params and "state" in params:
    code = params.get("code")
    state = params.get("state")
    if st.session_state.get("oauth_state") and st.session_state["oauth_state"] == state:
        try:
            tok = exchange_code_for_token(code)
            st.session_state["access_token"] = tok["access_token"]
            st.session_state["refresh_token"] = tok["refresh_token"]
            st.session_state["expires_at"] = tok["expires_at"]
            st.session_state["athlete"] = tok["athlete"]
            st.success("Свързване със Strava: ОК ✅")
        except Exception as e:
            st.error(f"Грешка при обмен на code → token: {e}")
        finally:
            # чист URL без параметри
            st.query_params.clear()
    else:
        st.error("Невалидно OAuth състояние (state). Опитай пак.")
        st.query_params.clear()

# Статус на сесията
logged_in = "access_token" in st.session_state

if not logged_in:
    st.write("Впиши се със Strava, за да синхронизираш активностите си към Supabase.")
    if st.button("Вход със Strava"):
        st.link_button("Продължи към Strava", oauth_login_url())
    st.stop()

# Рефреш токен ако е изтекъл
if datetime.now(timezone.utc) > st.session_state["expires_at"]:
    try:
        rt = refresh_access_token(st.session_state["refresh_token"])
        st.session_state["access_token"] = rt["access_token"]
        st.session_state["refresh_token"] = rt["refresh_token"]
        st.session_state["expires_at"] = rt["expires_at"]
    except Exception as e:
        st.error(f"Неуспешно опресняване на токена: {e}")
        st.stop()

athlete = st.session_state.get("athlete", {})
athlete_id = athlete.get("id")
st.info(f"Атлет: {athlete.get('firstname','')} {athlete.get('lastname','')}  •  ID: {athlete_id}")

col1, col2 = st.columns(2)
with col1:
    days = st.number_input("Изтегли последните (дни)", min_value=0, max_value=3650, value=30, step=1)
with col2:
    with_streams = st.checkbox("Записвай 1 Hz стриймове", value=True)

if st.button("Синхронизирай"):
    access_token = st.session_state["access_token"]
    after_ts = None
    if days > 0:
        after_ts = int(datetime.now(timezone.utc).timestamp() - days * 86400)

    try:
        acts = get_athlete_activities(access_token, after_ts=after_ts)
        st.write(f"Намерени активности: **{len(acts)}**")

        # 1) Upsert активности (ORM в db.py ще нормализира форматите)
        inserted = upsert_activities(acts)
        st.success(f"Обновени/вкарани активности: {inserted}")

        # 2) По желание – стриймове (chunk upsert вътре в db.py)
        if with_streams:
            pb = st.pro
