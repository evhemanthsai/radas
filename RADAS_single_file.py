#!/usr/bin/env python3
"""
RADAS - Road Accident Data Analysis System (single-file Flask app)
=================================================================
Python: Flask, Supabase Python SDK, Pandas, Matplotlib.
Database & authentication: Supabase (PostgreSQL + Supabase Auth).
UI: server-rendered HTML/CSS strings embedded in this Python file; no JS files.

SETUP:
  1. pip install "flask>=3.1,<4" "supabase>=2.20,<3" "pandas>=2,<3" "matplotlib>=3.8,<4" "cryptography>=42" "python-dotenv>=1"
  2. python RADAS_single_file.py --schema  > radas_setup.sql
     Paste the generated SQL into Supabase Dashboard > SQL Editor > Run.
  3. Run python RADAS_single_file_v1_3.py --init-env once to create .env
     next to the script. Edit SUPABASE_URL and SUPABASE_KEY in that file.
     To check configuration: python RADAS_single_file_v1_3.py --check-env
  4. Start the app: python RADAS_single_file_v1_3.py
  5. Open http://127.0.0.1:5000

CUSTOM CHOICES:
  Each of company, type and model selectors includes Other. Enter the custom
  text alongside that selector and click the next Load button. Custom values
  are saved in the existing Supabase text columns; no database migration needed.

Important: Never use a Supabase service_role/secret key for this application.
Use a publishable key (sb_publishable_...) or legacy anon key; RLS is enforced.
Auth tokens are encrypted before being held in Flask's signed browser cookie.
For production, move sessions to server-side storage, use HTTPS and rate limits.
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import sys
from collections import Counter
from datetime import date, timedelta
from functools import wraps
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: python-dotenv. Install it with: py -m pip install python-dotenv"
    ) from exc

# Always look beside this Python file, regardless of the terminal's current directory.
APP_DIR = Path(__file__).resolve().parent
DOTENV_PATH = APP_DIR / ".env"
if DOTENV_PATH.is_file():
    load_dotenv(dotenv_path=DOTENV_PATH, encoding="utf-8-sig")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from flask import (Flask, abort, flash, g, redirect, render_template,
                   request, session, url_for)
from jinja2 import DictLoader
from supabase import create_client

SQL = r"""
-- RADAS v1.1 -- execute ONCE in the Supabase SQL Editor.
-- NOTE: a purposefully anonymized reporting view bypasses owner RLS so all
-- authenticated users can see crowdsourced accident statistics, not owners.
create extension if not exists pgcrypto;

create table if not exists public.vehicles (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  person_name text not null check (length(trim(person_name)) between 2 and 100),
  company text not null,
  vehicle_type text not null,
  model_series text not null,
  registration_number text not null,
  purchase_date date not null,
  state text not null,
  city text not null,
  location text not null,
  pincode text not null check (pincode ~ '^[0-9]{6}$'),
  created_at timestamptz not null default now(),
  unique (user_id, registration_number)
);

create table if not exists public.accidents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  accident_date date not null,
  state text not null,
  city text not null,
  location text not null,
  severity text not null check (severity in ('Minor','Serious','Fatal')),
  injuries integer not null default 0 check (injuries >= 0),
  fatalities integer not null default 0 check (fatalities >= 0),
  details text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists public.accident_vehicles (
  accident_id uuid not null references public.accidents(id) on delete cascade,
  vehicle_id uuid not null references public.vehicles(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  primary key (accident_id, vehicle_id)
);

create index if not exists ix_vehicles_user on public.vehicles(user_id);
create index if not exists ix_accidents_user_date on public.accidents(user_id, accident_date);
create index if not exists ix_accidents_state_city on public.accidents(state, city);
create index if not exists ix_av_user on public.accident_vehicles(user_id);
create index if not exists ix_av_vehicle on public.accident_vehicles(vehicle_id);

alter table public.vehicles enable row level security;
alter table public.accidents enable row level security;
alter table public.accident_vehicles enable row level security;

revoke all on public.vehicles, public.accidents, public.accident_vehicles
  from anon, authenticated;
grant select, insert, update, delete
  on public.vehicles, public.accidents to authenticated;
grant select, insert, delete on public.accident_vehicles to authenticated;

-- Idempotent policy setup for classroom re-runs:
drop policy if exists vehicles_read on public.vehicles;
drop policy if exists vehicles_insert on public.vehicles;
drop policy if exists vehicles_update on public.vehicles;
drop policy if exists vehicles_delete on public.vehicles;
create policy vehicles_read on public.vehicles for select to authenticated
  using (user_id = (select auth.uid()));
create policy vehicles_insert on public.vehicles for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy vehicles_update on public.vehicles for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy vehicles_delete on public.vehicles for delete to authenticated
  using (user_id = (select auth.uid()));

drop policy if exists accidents_read on public.accidents;
drop policy if exists accidents_insert on public.accidents;
drop policy if exists accidents_update on public.accidents;
drop policy if exists accidents_delete on public.accidents;
create policy accidents_read on public.accidents for select to authenticated
  using (user_id = (select auth.uid()));
create policy accidents_insert on public.accidents for insert to authenticated
  with check (user_id = (select auth.uid()));
create policy accidents_update on public.accidents for update to authenticated
  using (user_id = (select auth.uid()))
  with check (user_id = (select auth.uid()));
create policy accidents_delete on public.accidents for delete to authenticated
  using (user_id = (select auth.uid()));

drop policy if exists av_read on public.accident_vehicles;
drop policy if exists av_insert on public.accident_vehicles;
drop policy if exists av_delete on public.accident_vehicles;
create policy av_read on public.accident_vehicles for select to authenticated
  using (user_id = (select auth.uid()));
create policy av_insert on public.accident_vehicles for insert to authenticated
  with check (
    user_id = (select auth.uid())
    and exists (select 1 from public.vehicles v
                where v.id=vehicle_id and v.user_id=(select auth.uid()))
    and exists (select 1 from public.accidents a
                where a.id=accident_id and a.user_id=(select auth.uid()))
  );
create policy av_delete on public.accident_vehicles for delete to authenticated
  using (user_id = (select auth.uid()));

-- Reporting VIEW only exposes accident, vehicle category and geography.
-- The view owner in Supabase SQL Editor is postgres; it reads across RLS.
-- Never include person_name, registration_number, user_id or personal emails.
create or replace view public.accident_analytics
  with (security_barrier=true) as
select a.id as accident_id,
       a.accident_date, a.state, a.city, a.location,
       a.severity, a.injuries, a.fatalities,
       v.company, v.vehicle_type, v.model_series
from public.accidents a
join public.accident_vehicles av on av.accident_id = a.id
join public.vehicles v on v.id = av.vehicle_id;
revoke all on public.accident_analytics from public, anon, authenticated;
grant select on public.accident_analytics to authenticated;
"""

STATES = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry"
)
TYPES = ("Car", "Motorcycle", "Scooter", "Auto Rickshaw", "Bus", "Truck", "Other")
CATALOG = {
    "Maruti Suzuki": {"Car": ["Swift", "Baleno", "Dzire", "Brezza", "Wagon R"]},
    "Tata Motors": {"Car": ["Nexon", "Punch", "Tiago", "Harrier"],
                    "Truck": ["Ace", "Signa"], "Bus": ["Starbus"]},
    "Hyundai": {"Car": ["i20", "Creta", "Venue", "Verna"]},
    "Mahindra": {"Car": ["Scorpio", "Thar", "XUV700", "Bolero"]},
    "Toyota": {"Car": ["Innova Crysta", "Fortuner", "Glanza"]},
    "Kia": {"Car": ["Seltos", "Sonet", "Carens"]},
    "Honda": {"Car": ["City", "Amaze", "Elevate"],
              "Motorcycle": ["Shine", "SP 125", "Unicorn"],
              "Scooter": ["Activa", "Dio"]},
    "Hero": {"Motorcycle": ["Splendor", "Passion", "Xtreme"],
             "Scooter": ["Pleasure", "Destini"]},
    "Bajaj": {"Motorcycle": ["Pulsar", "Platina", "Dominar"],
              "Auto Rickshaw": ["RE"]},
    "TVS": {"Motorcycle": ["Apache", "Raider", "Star City"],
            "Scooter": ["Jupiter", "Ntorq", "iQube"]},
    "Royal Enfield": {"Motorcycle": ["Classic 350", "Hunter 350", "Meteor 350"]},
    "Suzuki": {"Scooter": ["Access 125", "Burgman Street"],
               "Motorcycle": ["Gixxer", "V-Strom"]},
    "Ather": {"Scooter": ["450X", "Rizta"]},
    "Ola Electric": {"Scooter": ["S1 Pro", "S1 X"]},
    "Other": {t: ["Other / Custom"] for t in TYPES},
}
SEVERITIES = ("Minor", "Serious", "Fatal")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY") or secrets.token_urlsafe(48)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("HTTPS_ONLY", "0") == "1",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    MAX_CONTENT_LENGTH=64 * 1024,
)


def cipher():
    # Flask signs cookies; encrypting tokens prevents disclosure in cookie text.
    key = base64.urlsafe_b64encode(hashlib.sha256(app.secret_key.encode()).digest())
    return Fernet(key)


def encrypt_auth(access_token, refresh_token):
    data = json.dumps({"access": access_token, "refresh": refresh_token}).encode()
    return cipher().encrypt(data).decode()


def decrypt_auth(token):
    return json.loads(cipher().decrypt(token.encode()).decode())


def new_client():
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    if key.startswith("sb_secret_"):
        raise RuntimeError("Do NOT use a secret/service key. Configure a publishable/anon key.")
    return create_client(url, key)


def csrf():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def check_csrf():
    supplied = request.form.get("_csrf", "")
    if not supplied or not hmac.compare_digest(supplied, session.get("csrf", "")):
        abort(400, "Invalid CSRF token; reload the page and try again.")


@app.before_request
def load_identity():
    g.user = None
    g.sb = None
    encrypted = session.get("auth")
    if not encrypted:
        return
    try:
        tokens = decrypt_auth(encrypted)
        sb = new_client()  # Never share auth state across requests/users.
        auth_result = sb.auth.set_session(tokens["access"], tokens["refresh"])
        if not auth_result.session or not auth_result.user:
            raise ValueError("Session expired")
        fresh = auth_result.session
        if (fresh.access_token != tokens["access"] or
                fresh.refresh_token != tokens["refresh"]):
            session["auth"] = encrypt_auth(fresh.access_token, fresh.refresh_token)
        g.user = auth_result.user
        g.sb = sb
    except (InvalidToken, KeyError, ValueError):
        session.pop("auth", None)
    except Exception:
        # Avoid leaking internal auth/database error details to the browser.
        session.pop("auth", None)
        flash("Please sign in again to continue.", "error")


def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return func(*args, **kwargs)
    return wrapper


def user_id():
    return str(g.user.id)


def db_all(table, order="created_at", limit=5000):
    """Read account-owned rows. Paginates around PostgREST's 1,000-row cap."""
    rows = []
    for start in range(0, limit, 500):
        chunk = (g.sb.table(table).select("*")
                 .order(order).range(start, start + min(499, limit-start-1)).execute().data)
        rows.extend(chunk or [])
        if len(chunk or []) < 500:
            break
    return rows


def record(table, pk):
    rows = g.sb.table(table).select("*").eq("id", pk).limit(1).execute().data
    if not rows:
        abort(404, "Record not found or unavailable")
    return rows[0]


def clean_str(key, max_len=120, min_len=1):
    value = request.form.get(key, "").strip()
    if not min_len <= len(value) <= max_len:
        raise ValueError(f"{key.replace('_', ' ').title()} must be {min_len}-{max_len} characters")
    return value


def valid_date(value, label):
    try:
        parsed = date.fromisoformat(value)
    except Exception as exc:
        raise ValueError(f"Enter a valid {label}") from exc
    if parsed > date.today() or parsed.year < 1950:
        raise ValueError(f"{label} must be between 1950 and today")
    return parsed.isoformat()


def parse_vehicle():
    """Resolve preset or user-entered manufacturer/type/model safely."""
    brand_choice = clean_str("company", 80)
    type_choice = clean_str("vehicle_type", 40)
    model_choice = clean_str("model_series", 80)

    if brand_choice not in CATALOG:
        raise ValueError("Select a listed company or Other")
    if brand_choice == "Other":
        company = clean_str("custom_company", 80, 2)
        if company.casefold() == "other":
            raise ValueError("Enter your actual vehicle brand, not Other")
    else:
        company = brand_choice

    if type_choice == "Other":
        kind = clean_str("custom_type", 40, 2)
        if kind.casefold() == "other":
            raise ValueError("Enter your actual vehicle type, not Other")
    else:
        if type_choice not in TYPES or type_choice == "Other":
            raise ValueError("Select a valid vehicle type or Other")
        if brand_choice != "Other" and type_choice not in CATALOG[brand_choice]:
            raise ValueError("Choose Other to specify a custom type for this brand")
        kind = type_choice

    if model_choice in ("Other", "Other / Custom"):
        model = clean_str("custom_model", 80, 2)
        if model.casefold() in ("other", "other / custom"):
            raise ValueError("Enter your actual model/series, not Other")
    else:
        valid_catalog_models = CATALOG.get(brand_choice, {}).get(type_choice, [])
        if model_choice not in valid_catalog_models or model_choice == "Other / Custom":
            raise ValueError("Choose Other to specify a custom model/series")
        model = model_choice

    state = clean_str("state", 80)
    if state not in STATES:
        raise ValueError("Select an Indian state/UT")
    reg = clean_str("registration_number", 25, 4).upper()
    if not re.fullmatch(r"[A-Z0-9 /-]{4,25}", reg):
        raise ValueError("Registration/TR number contains invalid characters")
    pincode = clean_str("pincode", 6)
    if not re.fullmatch(r"[0-9]{6}", pincode):
        raise ValueError("Indian pincode must have exactly six digits")
    return {
        "person_name": clean_str("person_name", 100, 2),
        "company": company,
        "vehicle_type": kind,
        "model_series": model,
        "registration_number": reg,
        "purchase_date": valid_date(clean_str("purchase_date", 10), "purchase date"),
        "state": state,
        "city": clean_str("city", 80),
        "location": clean_str("location", 200),
        "pincode": pincode,
    }


def parse_accident(owned_vehicle_ids):
    ids = list(dict.fromkeys(request.form.getlist("vehicle_ids")))
    if not ids or len(ids) > 15 or not set(ids).issubset(owned_vehicle_ids):
        raise ValueError("Select between 1 and 15 of your registered vehicles")
    state = clean_str("state", 80)
    if state not in STATES:
        raise ValueError("Select a valid state/UT")
    severity = clean_str("severity", 20)
    if severity not in SEVERITIES:
        raise ValueError("Invalid severity")
    try:
        injuries = int(clean_str("injuries", 6))
        fatalities = int(clean_str("fatalities", 6))
    except ValueError as exc:
        raise ValueError("Injuries and fatalities must be whole numbers") from exc
    if not 0 <= injuries <= 10000 or not 0 <= fatalities <= 10000:
        raise ValueError("Injury/fatality counts must be between 0 and 10000")
    if severity == "Fatal" and fatalities == 0:
        raise ValueError("A fatal accident must include at least one fatality")
    if severity != "Fatal" and fatalities > 0:
        raise ValueError("Select Fatal severity when fatalities are recorded")
    return {
        "accident_date": valid_date(clean_str("accident_date", 10), "accident date"),
        "state": state,
        "city": clean_str("city", 80),
        "location": clean_str("location", 200),
        "severity": severity,
        "injuries": injuries,
        "fatalities": fatalities,
        "details": clean_str("details", 1500, 0),
    }, ids


def plot_to_base64(labels, values, title, kind="bar"):
    if not labels:
        return ""
    fig, ax = plt.subplots(figsize=(8.4, 3.5), dpi=110)
    try:
        if kind == "line":
            ax.plot(range(len(labels)), values, marker="o", linewidth=2.4, color="#0e7490")
            ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        elif kind == "pie":
            ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90)
        else:
            indices = list(range(len(labels)))
            ax.barh(indices, values, color="#2563eb")
            ax.set_yticks(indices, labels)
            ax.invert_yaxis()
            for i, v in enumerate(values):
                ax.text(v, i, f"  {int(v)}", va="center", fontsize=9)
            ax.set_xlim(0, max(values) * 1.20 + 0.5)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
        if kind != "pie":
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="x" if kind == "bar" else "y", alpha=0.18)
        fig.tight_layout()
        output = io.BytesIO()
        fig.savefig(output, format="png", bbox_inches="tight", facecolor="white")
        return base64.b64encode(output.getvalue()).decode("ascii")
    finally:
        plt.close(fig)


BASE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} | RADAS</title>
<style>
:root{color-scheme:light;--bg:#f3f6fb;--ink:#13213a;--muted:#627188;--line:#dee5ef;--blue:#245de7;--white:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
a{color:var(--blue);text-decoration:none}.top{background:#0e1930;color:#fff;display:flex;align-items:center;justify-content:space-between;gap:16px;padding:15px max(18px,calc((100vw - 1180px)/2))}
.top a{color:#e5eefe;margin-right:18px}.brand{font-size:22px;font-weight:900;letter-spacing:.7px}.top nav{display:flex;align-items:center;flex-wrap:wrap;gap:8px}.top nav a{margin:0;padding:7px 9px;border-radius:7px}.top nav a:hover{background:#213453}
main{max-width:1180px;margin:26px auto;padding:0 20px 40px}h1{font-size:30px;line-height:1.18;margin:0 0 10px}h2{font-size:19px;margin:0 0 13px}h3{font-size:15px;margin:0 0 10px}.muted{color:var(--muted)}
.card{background:var(--white);border:1px solid var(--line);border-radius:15px;padding:22px;box-shadow:0 2px 7px #0c193008;margin-bottom:18px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.kpi b{font-size:33px;display:block;margin-top:3px}.kpi small{color:var(--muted)}
label{display:block;font-weight:650;font-size:13px;margin:0 0 5px}input,select,textarea{width:100%;border:1px solid #c9d4e2;border-radius:9px;padding:11px 12px;background:#fff;font:inherit;color:var(--ink)}input:focus,select:focus,textarea:focus{outline:2px solid #94bdff;outline-offset:1px}textarea{min-height:86px}select[multiple]{min-height:130px}
.field{margin-bottom:15px}.formgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:0 17px}.btn{display:inline-block;border:0;background:var(--blue);color:white!important;border-radius:9px;padding:10px 17px;font-weight:650;font:inherit;cursor:pointer}.btn.ghost{background:#e9efff;color:#2447a5!important}.btn.danger{background:#b91c1c}.btn.small{padding:6px 9px;font-size:12px}.btn:hover{filter:brightness(.94)}.actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px}
.alert{padding:13px 16px;border-radius:10px;margin-bottom:14px;background:#e0f2fe;color:#0c4a6e}.alert.error{background:#fef0f0;color:#991b1b}.hint{background:#eff6ff;border-left:3px solid #2563eb;padding:11px 14px;color:#294368;font-size:13px;margin:14px 0}
.tablewrap{overflow:auto}table{width:100%;border-collapse:collapse;background:white;font-size:13px}th,td{text-align:left;border-bottom:1px solid var(--line);padding:11px 10px;vertical-align:top}th{font-weight:750;color:#42516a;background:#f8fafc}td .actions{margin:0}.chart img{width:100%;height:auto;display:block}.hero{padding:38px 26px;background:linear-gradient(115deg,#14294b,#294db2);color:white;border-radius:16px;margin-bottom:22px}.hero p{max-width:680px;color:#d9e6ff}.tag{background:#e9efff;color:#274e9d;border-radius:5px;font-size:12px;padding:3px 7px}.subnav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:17px}.subnav a{padding:9px 12px;background:#fff;border:1px solid var(--line);border-radius:8px}.empty{text-align:center;padding:35px;color:var(--muted)}footer{text-align:center;color:var(--muted);font-size:12px;margin:25px}
@media(max-width:760px){.grid,.grid3,.formgrid{grid-template-columns:1fr}h1{font-size:25px}.top{flex-direction:column;align-items:flex-start}.card{padding:17px}main{margin:15px auto}.top nav a{padding:5px 6px}}
</style></head><body>
<header class="top"><div class="brand">RADAS <span style="font-weight:400;font-size:12px;color:#94aad1">Flask · Supabase</span></div><nav>
<a href="{{ url_for('dashboard') }}">Dashboard</a>
{% if g.user %}<a href="{{ url_for('vehicle_form') }}">Register Vehicle</a><a href="{{ url_for('crud') }}">CRUD</a><a href="{{ url_for('reports') }}">Reports</a>
<form action="{{url_for('logout')}}" method="post" style="display:inline;margin:0"><input type="hidden" name="_csrf" value="{{csrf()}}"><button class="btn small ghost" type="submit">Logout</button></form>
{% else %}<a href="{{url_for('login')}}">Login</a><a href="{{url_for('signup')}}">Sign Up</a>{% endif %}</nav></header>
<main>{% for category,message in get_flashed_messages(with_categories=true) %}<div class="alert {{category}}">{{message}}</div>{% endfor %}
{% block content %}{% endblock %}</main><footer>RADAS · Educational project · Data is self-reported and not verified official accident statistics.</footer></body></html>"""

DASHBOARD = """{% extends 'base.html' %}{% block content %}
{% if not g.user %}
<div class="hero"><span class="tag">ROAD SAFETY ANALYTICS</span><h1 style="margin-top:12px">Road Accident Data Analysis System</h1>
<p>Register vehicles, record accident incidents, manage records and explore accident patterns by state, city and vehicle model.</p>
<div class="actions"><a class="btn" href="{{url_for('signup')}}">Create Account</a><a class="btn ghost" href="{{url_for('login')}}">Login</a></div></div>
<div class="grid3"><div class="card"><h2>01 · Registration</h2><p class="muted">Person, vehicle, purchase and location details saved in Supabase.</p></div>
<div class="card"><h2>02 · Record Management</h2><p class="muted">Create, view, edit and delete your vehicles and accident incidents.</p></div>
<div class="card"><h2>03 · Analytics</h2><p class="muted">Filter cities, states, makes and models and visualize with Matplotlib.</p></div></div>
{% else %}
<h1>Dashboard</h1><p class="muted">Welcome, {{ g.user.email }}. Manage your records and explore shared accident analytics.</p>
<div class="grid3"><div class="card kpi"><small>Your registered vehicles</small><b>{{ vehicle_count }}</b></div>
<div class="card kpi"><small>Your accident incidents</small><b>{{ accident_count }}</b></div>
<div class="card kpi"><small>Available modules</small><b>3</b></div></div>
<div class="card"><h2>Quick actions</h2><div class="actions"><a href="{{url_for('vehicle_form')}}" class="btn">+ Register Vehicle</a>
<a href="{{url_for('accident_form')}}" class="btn ghost">+ Report Accident</a><a href="{{url_for('crud')}}" class="btn ghost">Manage CRUD</a><a href="{{url_for('reports')}}" class="btn ghost">View Reports</a></div></div>
<div class="hint">Accident counts are based on manually entered incidents, not automatically inferred from vehicle registrations.</div>
{% endif %}{% endblock %}"""

AUTH = """{% extends 'base.html' %}{% block content %}<div class="card" style="max-width:470px;margin:42px auto">
<h1>{{ 'Create an account' if signup else 'Welcome back' }}</h1>
<p class="muted">{{ 'Use an email address and a secure password.' if signup else 'Sign in to manage vehicles and accidents.' }}</p>
<form method="post"><input name="_csrf" type="hidden" value="{{csrf()}}">
<div class="field"><label>Email</label><input name="email" type="email" required maxlength="200" autocomplete="email"></div>
<div class="field"><label>Password</label><input name="password" type="password" required minlength="8" maxlength="128" autocomplete="{{'new-password' if signup else 'current-password'}}"></div>
<button class="btn" type="submit">{{'Sign Up' if signup else 'Login'}}</button></form>
<p class="muted">{% if signup %}Have an account? <a href="{{url_for('login')}}">Login</a>
{% else %}New user? <a href="{{url_for('signup')}}">Create an account</a>{% endif %}</p></div>{% endblock %}"""

VEHICLE_FORM = """{% extends 'base.html' %}{% block content %}
<h1>{{'Edit Vehicle' if existing else 'Register Vehicle'}}</h1>
<p class="muted">Select a company, type, and model. Select <b>Other</b> at any step to enter a custom value. Click each Load button to update the next list (Python-only, no JavaScript).</p>
<div class="card">
<h2>Step 1 · Vehicle company / brand</h2>
<form method="get" action="{{url_for('vehicle_form',pk=existing.id) if existing else url_for('vehicle_form')}}">
<div class="formgrid">
<div class="field"><label for="manufacturer">Vehicle Company</label>
<select id="manufacturer" name="company" required>
{% for item in companies %}<option value="{{item}}" {{'selected' if item==chosen_company else ''}}>{{item}}</option>{% endfor %}
</select></div>
<div class="field"><label for="custom_company">Custom Company / Brand (if Other selected)</label>
<input id="custom_company" name="custom_company" maxlength="80" placeholder="e.g. Rivian, Local Motors" value="{{custom_company_value}}"></div>
</div>
<button class="btn ghost small" type="submit">Load Available Types</button>
</form>
<hr style="border:0;border-top:1px solid #e2e8f0;margin:18px 0">
<h2>Step 2 · Vehicle type</h2>
<form method="get" action="{{url_for('vehicle_form',pk=existing.id) if existing else url_for('vehicle_form')}}">
<input type="hidden" name="company" value="{{chosen_company}}">
<input type="hidden" name="custom_company" value="{{custom_company_value}}">
<div class="formgrid">
<div class="field"><label for="vehicle_type">Vehicle Type — {{brand_display}}</label>
<select id="vehicle_type" name="vehicle_type" required>
{% for item in available_types %}<option value="{{item}}" {{'selected' if item==chosen_type else ''}}>{{item}}</option>{% endfor %}
</select></div>
<div class="field"><label for="custom_type">Custom Vehicle Type (if Other selected)</label>
<input id="custom_type" name="custom_type" maxlength="40" placeholder="e.g. Electric Tricycle" value="{{custom_type_value}}"></div>
</div>
<button class="btn ghost small" type="submit">Load Vehicle Models / Series</button>
</form>
<div class="hint"><b>Selected:</b> {{brand_display}} → {{type_display}}. Available models: {{models|join(', ')}}.</div>
<hr style="border:0;border-top:1px solid #e2e8f0;margin:18px 0">
<h2>Step 3 · Vehicle model and owner details</h2>
<form method="post" action="{{url_for('vehicle_form',pk=existing.id) if existing else url_for('vehicle_form')}}">
<input type="hidden" name="_csrf" value="{{csrf()}}">
<input type="hidden" name="company" value="{{chosen_company}}"><input type="hidden" name="custom_company" value="{{custom_company_value}}">
<input type="hidden" name="vehicle_type" value="{{chosen_type}}"><input type="hidden" name="custom_type" value="{{custom_type_value}}">
<p><span class="tag">{{brand_display}} / {{type_display}}</span></p>
<div class="formgrid">
<div class="field"><label>Person Name</label><input name="person_name" required maxlength="100" value="{{v.person_name|default('',true)}}"></div>
<div class="field"><label>Vehicle Name / Series</label><select name="model_series" required>
{% for item in models %}<option value="{{item}}" {{'selected' if item==chosen_model else ''}}>{{item}}</option>{% endfor %}</select></div>
<div class="field"><label>Custom Model / Series (if Other selected)</label><input name="custom_model" maxlength="80" placeholder="e.g. Model X EV 2026" value="{{custom_model_value}}"></div>
<div class="field"><label>Registration Number / TR Number</label><input name="registration_number" required maxlength="25" value="{{v.registration_number|default('',true)}}"></div>
<div class="field"><label>Date of Purchase</label><input name="purchase_date" type="date" required max="{{today}}" value="{{v.purchase_date|default('',true)}}"></div>
<div class="field"><label>State / UT</label><select name="state" required><option value="">Select State</option>
{% for item in states %}<option value="{{item}}" {{'selected' if item==v.state else ''}}>{{item}}</option>{% endfor %}</select></div>
<div class="field"><label>City</label><input name="city" required maxlength="80" value="{{v.city|default('',true)}}"></div>
<div class="field"><label>Location / Area</label><input name="location" required maxlength="200" value="{{v.location|default('',true)}}"></div>
<div class="field"><label>Pincode</label><input name="pincode" required inputmode="numeric" pattern="[0-9]{6}" maxlength="6" value="{{v.pincode|default('',true)}}"></div>
</div><div class="actions"><button class="btn" type="submit">{{'Save Changes' if existing else 'Save to Supabase'}}</button>
<a class="btn ghost" href="{{url_for('crud')}}">Cancel</a></div></form>
</div>{% endblock %}"""

ACCIDENT_FORM = """{% extends 'base.html' %}{% block content %}
<h1>{{'Edit Accident' if existing else 'Report Accident'}}</h1>
<p class="muted">Link each incident to one or more of your registered vehicles. Data is self-reported.</p>
<div class="card">{% if not vehicles %}<div class="hint">Register a vehicle first. <a href="{{url_for('vehicle_form')}}">Register vehicle</a>.</div>
{% else %}<form method="post"><input type="hidden" name="_csrf" value="{{csrf()}}"><div class="formgrid">
<div class="field"><label>Vehicles involved (Ctrl / Cmd + click for multiple)</label>
<select name="vehicle_ids" multiple size="6" required>{% for x in vehicles %}<option value="{{x.id}}" {{'selected' if x.id in selected_ids else ''}}>{{x.registration_number}} — {{x.company}} {{x.model_series}}</option>{% endfor %}</select></div>
<div class="field"><label>Accident Date</label><input name="accident_date" type="date" required max="{{today}}" value="{{a.accident_date|default('',true)}}"></div>
<div class="field"><label>State / UT</label><select name="state" required><option value="">Select State</option>
{% for x in states %}<option value="{{x}}" {{'selected' if x==a.state else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>City</label><input name="city" required maxlength="80" value="{{a.city|default('',true)}}"></div>
<div class="field"><label>Accident Location / Road</label><input name="location" required maxlength="200" value="{{a.location|default('',true)}}"></div>
<div class="field"><label>Accident Severity</label><select name="severity" required>{% for x in severities %}<option value="{{x}}" {{'selected' if x==a.severity else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>Number of Injured People</label><input name="injuries" type="number" min="0" max="10000" required value="{{a.injuries|default(0)}}"></div>
<div class="field"><label>Number of Fatalities</label><input name="fatalities" type="number" min="0" max="10000" required value="{{a.fatalities|default(0)}}"></div>
</div><div class="field"><label>Incident Details (optional)</label><textarea name="details" maxlength="1500">{{a.details|default('',true)}}</textarea></div>
<div class="actions"><button class="btn" type="submit">{{'Update Incident' if existing else 'Save Accident Record'}}</button><a class="btn ghost" href="{{url_for('crud')}}">Cancel</a></div></form>{% endif %}</div>{% endblock %}"""

CRUD = """{% extends 'base.html' %}{% block content %}<h1>CRUD Management</h1><p class="muted">Only your own personal records can be edited or deleted.</p>
<div class="actions" style="margin-bottom:20px"><a href="{{url_for('vehicle_form')}}" class="btn">+ New Vehicle</a><a href="{{url_for('accident_form')}}" class="btn ghost">+ New Accident</a></div>
<div class="card"><h2>Registered Vehicles ({{vehicles|length}})</h2>
<div class="tablewrap"><table><thead><tr><th>Owner</th><th>Company / Model</th><th>Type</th><th>Registration / TR</th><th>City / State</th><th>Actions</th></tr></thead><tbody>
{% for v in vehicles %}<tr><td>{{v.person_name}}</td><td>{{v.company}}<br><span class="muted">{{v.model_series}}</span></td><td>{{v.vehicle_type}}</td>
<td>{{v.registration_number}}</td><td>{{v.city}}, {{v.state}}</td><td><div class="actions"><a class="btn small ghost" href="{{url_for('vehicle_form',pk=v.id)}}">Edit</a>
<details><summary style="cursor:pointer;color:#b91c1c">Delete</summary><form method="post" action="{{url_for('delete_vehicle',pk=v.id)}}"><input type="hidden" name="_csrf" value="{{csrf()}}"><button class="btn danger small">Confirm</button></form></details></div></td></tr>
{% else %}<tr><td colspan="6" class="empty">No vehicles yet.</td></tr>{% endfor %}</tbody></table></div></div>
<div class="card"><h2>Accident Incidents ({{accidents|length}})</h2>
<div class="tablewrap"><table><thead><tr><th>Date</th><th>Location</th><th>Severity</th><th>Injuries / Fatalities</th><th>Vehicles Linked</th><th>Actions</th></tr></thead><tbody>
{% for a in accidents %}<tr><td>{{a.accident_date}}</td><td>{{a.location}}<br><span class="muted">{{a.city}}, {{a.state}}</span></td>
<td>{{a.severity}}</td><td>{{a.injuries}} / {{a.fatalities}}</td><td>{{links.get(a.id,0)}}</td>
<td><div class="actions"><a class="btn small ghost" href="{{url_for('accident_form',pk=a.id)}}">Edit</a>
<details><summary style="cursor:pointer;color:#b91c1c">Delete</summary><form method="post" action="{{url_for('delete_accident',pk=a.id)}}"><input type="hidden" name="_csrf" value="{{csrf()}}"><button class="btn danger small">Confirm</button></form></details></div></td></tr>
{% else %}<tr><td colspan="6" class="empty">No accidents yet.</td></tr>{% endfor %}</tbody></table></div></div>{% endblock %}"""

REPORTS = """{% extends 'base.html' %}{% block content %}<h1>Accident Analytics Reports</h1>
<p class="muted">Anonymized shared data from user-entered incidents. Charts generated by Python / Matplotlib.</p>
<div class="hint"><strong>Interpretation:</strong> One accident can involve several vehicles. State/city totals count unique incidents; model charts count incidents involving that model. These are not verified government records.</div>
<div class="card"><h2>Filters</h2><form method="get" class="formgrid">
<div class="field"><label>State</label><select name="state"><option value="">All States</option>{% for x in options.states %}<option value="{{x}}" {{'selected' if f.state==x else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>City (choose state, Apply Filters, then choose city)</label><select name="city"><option value="">All Cities</option>{% for x in options.cities %}<option value="{{x}}" {{'selected' if f.city==x else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>Vehicle Company</label><select name="company"><option value="">All Companies</option>{% for x in options.companies %}<option value="{{x}}" {{'selected' if f.company==x else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>Vehicle Type</label><select name="vehicle_type"><option value="">All Types</option>{% for x in options.types %}<option value="{{x}}" {{'selected' if f.vehicle_type==x else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>Vehicle Model / Series</label><select name="model_series"><option value="">All Models</option>{% for x in options.models %}<option value="{{x}}" {{'selected' if f.model_series==x else ''}}>{{x}}</option>{% endfor %}</select></div>
<div class="field"><label>From Date</label><input type="date" name="start" value="{{f.start}}"></div>
<div class="field"><label>To Date</label><input type="date" name="end" value="{{f.end}}"></div>
<div class="actions"><button class="btn" type="submit">Apply Filters</button><a class="btn ghost" href="{{url_for('reports')}}">Reset</a></div></form></div>
<div class="grid3"><div class="card kpi"><small>Accident Incidents</small><b>{{summary.accidents}}</b></div>
<div class="card kpi"><small>Vehicle Involvements</small><b>{{summary.involvements}}</b></div>
<div class="card kpi"><small>Fatalities (incident-level)</small><b>{{summary.fatalities}}</b></div></div>
{% if limited %}<div class="alert error">Only the first {{cap}} reporting rows were loaded. Narrow the filters or use a database-side analytics function for larger datasets.</div>{% endif %}
{% if summary.accidents %}<div class="grid">{% for title,img in charts.items() %}<div class="card chart"><img src="data:image/png;base64,{{img}}" alt="{{title}} chart"></div>{% endfor %}</div>
<div class="card"><h2>Top Locations</h2><div class="tablewrap"><table><thead><tr><th>State</th><th>City</th><th>Distinct Incidents</th></tr></thead><tbody>
{% for x in locations %}<tr><td>{{x.state}}</td><td>{{x.city}}</td><td>{{x.count}}</td></tr>{% endfor %}</tbody></table></div></div>
{% else %}<div class="card empty">No accident records match these filters. Add records from the CRUD page or adjust your filters.</div>{% endif %}
<div class="card"><h2>Check Accidents by Your Vehicle Registration Number</h2>
<p class="muted">Private lookup: only registration numbers owned by your account can be searched.</p>
<form method="get" action="{{url_for('vehicle_lookup')}}" class="formgrid"><div class="field"><label>Registration Number / TR Number</label>
<input name="registration" required maxlength="25" placeholder="e.g. AP03AB1234" value="{{lookup_reg}}"></div>
<div class="field" style="align-self:end"><button class="btn" type="submit">Check Accidents</button></div></form>
{% if lookup is not none %}<div class="hint"><b>{{lookup|length}}</b> accident(s) found for this registration in your account.</div>
<div class="tablewrap"><table><thead><tr><th>Date</th><th>State</th><th>City</th><th>Location</th><th>Severity</th></tr></thead><tbody>
{% for x in lookup %}<tr><td>{{x.accident_date}}</td><td>{{x.state}}</td><td>{{x.city}}</td><td>{{x.location}}</td><td>{{x.severity}}</td></tr>{% endfor %}
</tbody></table></div>{% endif %}</div>
{% endblock %}"""

app.jinja_loader = DictLoader({"base.html": BASE, "dashboard.html": DASHBOARD,
                               "auth.html": AUTH, "vehicle.html": VEHICLE_FORM,
                               "accident.html": ACCIDENT_FORM, "crud.html": CRUD,
                               "reports.html": REPORTS})
app.jinja_env.globals["csrf"] = csrf


@app.route("/")
def dashboard():
    vehicles = accidents = 0
    if g.user:
        try:
            vehicles = len(db_all("vehicles"))
            accidents = len(db_all("accidents"))
        except Exception:
            flash("Could not load dashboard statistics. Check your Supabase schema/RLS.", "error")
    return render_template("dashboard.html", title="Dashboard", vehicle_count=vehicles,
                           accident_count=accidents)


def auth_handler(signup):
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        check_csrf()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if len(email) > 200 or "@" not in email or not 8 <= len(password) <= 128:
            flash("Enter a valid email and password (8-128 characters).", "error")
        else:
            try:
                sb = new_client()
                if signup:
                    result = sb.auth.sign_up({"email": email, "password": password})
                    if not result.session:
                        flash("Account requested. Check your email for verification, then log in.")
                        return redirect(url_for("login"))
                else:
                    result = sb.auth.sign_in_with_password({"email": email, "password": password})
                if not result.session:
                    raise ValueError("No authentication session returned")
                session.clear()
                session.permanent = True
                session["auth"] = encrypt_auth(result.session.access_token,
                                               result.session.refresh_token)
                csrf()
                flash("Signed in successfully.")
                return redirect(url_for("dashboard"))
            except Exception:
                flash("Authentication failed. Check your credentials, email verification, or Supabase settings.", "error")
    return render_template("auth.html", title="Sign Up" if signup else "Login", signup=signup)


@app.route("/login", methods=["GET", "POST"])
def login():
    return auth_handler(False)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    return auth_handler(True)


@app.post("/logout")
def logout():
    check_csrf()
    if g.sb:
        try:
            g.sb.auth.sign_out()
        except Exception:
            pass
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("dashboard"))


@app.route("/vehicle/new", methods=["GET", "POST"])
@app.route("/vehicle/<uuid:pk>/edit", methods=["GET", "POST"])
@login_required
def vehicle_form(pk=None):
    existing = record("vehicles", str(pk)) if pk else None
    form = request.values
    v = dict(existing or {})
    # After a validation error, preserve submitted fields.
    if request.method == "POST":
        v.update(request.form.to_dict())
        check_csrf()
        try:
            payload = parse_vehicle()
            if existing:
                g.sb.table("vehicles").update(payload).eq("id", str(pk)).execute()
                flash("Vehicle updated successfully.")
            else:
                g.sb.table("vehicles").insert({**payload, "user_id": user_id()}).execute()
                flash("Vehicle registered successfully in Supabase.")
            return redirect(url_for("crud"))
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception:
            flash("Unable to save vehicle. Check duplicate registrations and table permissions.", "error")
    # Three cascading Python-only choices, each with an Other/custom path.
    # For edit, map any unlisted stored values back to Other + its text field.
    existing_brand = (existing or {}).get("company", "")
    company_value = form.get("company")
    if company_value is None:
        chosen_company = existing_brand if existing_brand in CATALOG else ("Other" if existing_brand else "Maruti Suzuki")
    else:
        chosen_company = company_value if company_value in CATALOG else "Other"

    if "custom_company" in form:
        custom_company_value = form.get("custom_company", "").strip()
    elif chosen_company == "Other" and existing_brand not in CATALOG:
        custom_company_value = existing_brand
    else:
        custom_company_value = ""
    brand_display = custom_company_value if chosen_company == "Other" else chosen_company
    same_brand = bool(existing) and brand_display == existing_brand

    # Fixed company: curated types + Other; custom company: every standard type + Other.
    if chosen_company == "Other":
        available_types = [k for k in TYPES if k != "Other"] + ["Other"]
    else:
        available_types = [k for k in TYPES if k != "Other" and k in CATALOG[chosen_company]] + ["Other"]

    type_value = form.get("vehicle_type")
    if type_value is None:
        existing_type = (existing or {}).get("vehicle_type", "") if same_brand else ""
        chosen_type = existing_type if existing_type in available_types else ("Other" if existing_type else available_types[0])
    else:
        chosen_type = type_value if type_value in available_types else "Other"

    if "custom_type" in form:
        custom_type_value = form.get("custom_type", "").strip()
    elif chosen_type == "Other" and same_brand and existing.get("vehicle_type") not in available_types:
        custom_type_value = existing["vehicle_type"]
    else:
        custom_type_value = ""
    type_display = custom_type_value if chosen_type == "Other" else chosen_type
    same_type = same_brand and type_display == (existing or {}).get("vehicle_type")

    # Unknown brand/type has no preloaded models; Other is always available.
    catalog_models = (CATALOG.get(chosen_company, {}).get(chosen_type, [])
                      if chosen_company != "Other" and chosen_type != "Other" else [])
    models = [m for m in catalog_models if m not in ("Other", "Other / Custom")]
    models.append("Other")

    model_value = form.get("model_series")
    if model_value is None:
        existing_model = (existing or {}).get("model_series", "") if same_type else ""
        chosen_model = existing_model if existing_model in models else ("Other" if existing_model else models[0])
    else:
        chosen_model = model_value if model_value in models else "Other"

    if "custom_model" in form:
        custom_model_value = form.get("custom_model", "").strip()
    elif same_type and (existing or {}).get("model_series", "") not in models:
        custom_model_value = existing["model_series"]
    else:
        custom_model_value = ""

    return render_template("vehicle.html", title="Vehicle Registration",
                           v=v, existing=existing, companies=sorted(CATALOG),
                           available_types=available_types,
                           chosen_company=chosen_company,
                           custom_company_value=custom_company_value,
                           brand_display=brand_display,
                           chosen_type=chosen_type,
                           custom_type_value=custom_type_value,
                           type_display=type_display,
                           chosen_model=chosen_model, models=models,
                           custom_model_value=custom_model_value, states=STATES,
                           today=date.today().isoformat())


@app.route("/accident/new", methods=["GET", "POST"])
@app.route("/accident/<uuid:pk>/edit", methods=["GET", "POST"])
@login_required
def accident_form(pk=None):
    existing = record("accidents", str(pk)) if pk else None
    vehicles = db_all("vehicles")
    owned = {v["id"] for v in vehicles}
    selected_ids = set()
    if existing:
        old_links = (g.sb.table("accident_vehicles").select("vehicle_id")
                     .eq("accident_id", str(pk)).execute().data or [])
        selected_ids = {x["vehicle_id"] for x in old_links}
    a = dict(existing or {})
    if request.method == "POST":
        check_csrf()
        a.update(request.form.to_dict())
        selected_ids = set(request.form.getlist("vehicle_ids"))
        try:
            payload, ids = parse_accident(owned)
            if existing:
                # Non-atomic demo update. In production implement this inside a
                # Postgres function / RPC to guarantee an all-or-nothing save.
                g.sb.table("accidents").update(payload).eq("id", str(pk)).execute()
                current = {x["vehicle_id"] for x in old_links}
                for remove_id in current - set(ids):
                    (g.sb.table("accident_vehicles").delete()
                     .eq("accident_id", str(pk)).eq("vehicle_id", remove_id).execute())
                if set(ids) - current:
                    g.sb.table("accident_vehicles").insert([
                        {"accident_id": str(pk), "vehicle_id": vid, "user_id": user_id()}
                        for vid in set(ids) - current
                    ]).execute()
                flash("Accident incident updated.")
            else:
                created = (g.sb.table("accidents")
                           .insert({**payload, "user_id": user_id()}).execute().data or [])
                if not created:
                    raise RuntimeError("Insert returned no row")
                acc_id = created[0]["id"]
                try:
                    g.sb.table("accident_vehicles").insert([
                        {"accident_id": acc_id, "vehicle_id": vid, "user_id": user_id()}
                        for vid in ids
                    ]).execute()
                except Exception:
                    g.sb.table("accidents").delete().eq("id", acc_id).execute()
                    raise
                flash("Accident incident saved successfully.")
            return redirect(url_for("crud"))
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception:
            flash("Could not save accident. Please confirm linked vehicles and Supabase RLS.", "error")
    return render_template("accident.html", title="Accident Entry", existing=existing,
                           vehicles=vehicles, selected_ids=selected_ids, a=a,
                           states=STATES, severities=SEVERITIES,
                           today=date.today().isoformat())


@app.route("/crud")
@login_required
def crud():
    try:
        vehicles = db_all("vehicles")
        accidents = db_all("accidents")
        link_rows = db_all("accident_vehicles", "accident_id")
    except Exception:
        flash("Unable to load records. Make sure setup SQL was executed.", "error")
        vehicles, accidents, link_rows = [], [], []
    links = Counter(r["accident_id"] for r in link_rows)
    return render_template("crud.html", title="CRUD", vehicles=vehicles,
                           accidents=accidents, links=links)


@app.post("/vehicle/<uuid:pk>/delete")
@login_required
def delete_vehicle(pk):
    check_csrf()
    record("vehicles", str(pk))
    try:
        # Prevent orphaned incidents that would otherwise vanish from reports.
        attached = (g.sb.table("accident_vehicles").select("accident_id")
                    .eq("vehicle_id", str(pk)).limit(1).execute().data or [])
        if attached:
            flash("Remove this vehicle from linked incidents (or delete those incidents) first.", "error")
            return redirect(url_for("crud"))
        g.sb.table("vehicles").delete().eq("id", str(pk)).execute()
        flash("Vehicle deleted. Linked incident associations were removed.")
    except Exception:
        flash("Unable to delete vehicle.", "error")
    return redirect(url_for("crud"))


@app.post("/accident/<uuid:pk>/delete")
@login_required
def delete_accident(pk):
    check_csrf()
    record("accidents", str(pk))
    try:
        g.sb.table("accidents").delete().eq("id", str(pk)).execute()
        flash("Accident record deleted.")
    except Exception:
        flash("Unable to delete accident.", "error")
    return redirect(url_for("crud"))


def analytics_rows(cap=10000):
    """Retrieve de-identified shared report rows. Never fetch names or plates."""
    items = []
    for offset in range(0, cap + 1, 500):
        page = (g.sb.table("accident_analytics").select("*")
                .order("accident_id")
                .range(offset, offset + 499).execute().data or [])
        items.extend(page)
        if len(page) < 500:
            break
    limited = len(items) > cap
    return items[:cap], limited


def report_payload(df):
    if df.empty:
        return {"accidents": 0, "involvements": 0, "fatalities": 0}, {}, []
    unique = df.drop_duplicates("accident_id")
    summary = {"accidents": len(unique), "involvements": len(df),
               "fatalities": int(pd.to_numeric(unique["fatalities"]).sum())}
    def top_count(group, field, n=8):
        rows = group.groupby(field, dropna=False)["accident_id"].nunique().nlargest(n)
        return [str(x) for x in rows.index], [int(x) for x in rows.values]
    charts = {}
    state_labels, state_values = top_count(unique, "state")
    city_labels, city_values = top_count(unique, "city")
    model_df = df.assign(model=df["company"].astype(str)+" · "+df["model_series"].astype(str))
    model_labels, model_values = top_count(model_df, "model")
    months = unique.copy()
    months["month"] = pd.to_datetime(months["accident_date"]).dt.strftime("%Y-%m")
    timeline = months.groupby("month")["accident_id"].nunique().sort_index()
    severity = unique.groupby("severity")["accident_id"].nunique().sort_values(ascending=False)
    locations = (unique.groupby(["state", "city"], dropna=False)["accident_id"]
                 .nunique().sort_values(ascending=False).head(12))
    charts["Accidents by State"] = plot_to_base64(state_labels, state_values, "Accidents by State")
    charts["Accidents by City"] = plot_to_base64(city_labels, city_values, "Accidents by City")
    charts["Vehicle Model Analysis"] = plot_to_base64(model_labels, model_values, "Accidents by Vehicle Model")
    charts["Monthly Accident Trend"] = plot_to_base64(list(timeline.index), list(timeline.values), "Monthly Trends", "line")
    charts["Accident Severity"] = plot_to_base64(list(severity.index), list(severity.values), "Severity Distribution", "pie")
    loc_out = [{"state": a, "city": b, "count": int(v)} for (a, b), v in locations.items()]
    return summary, charts, loc_out


def private_lookup(reg):
    # Query scoped by RLS: other owners' registration numbers cannot be searched.
    matches = (g.sb.table("vehicles").select("id")
               .eq("registration_number", reg.upper()).limit(1).execute().data or [])
    if not matches:
        return []
    vehicle_id = matches[0]["id"]
    links = (g.sb.table("accident_vehicles").select("accident_id")
             .eq("vehicle_id", vehicle_id).execute().data or [])
    ids = list({x["accident_id"] for x in links})
    if not ids:
        return []
    # Batch in case a vehicle has hundreds of incidents.
    events = []
    for start in range(0, len(ids), 100):
        batch = ids[start:start+100]
        events += (g.sb.table("accidents")
                   .select("accident_date,state,city,location,severity")
                   .in_("id", batch).order("accident_date", desc=True).execute().data or [])
    return sorted(events, key=lambda r: r["accident_date"], reverse=True)


@app.get("/reports")
@login_required
def reports():
    try:
        raw, limited = analytics_rows()
    except Exception:
        raw, limited = [], False
        flash("Could not load analytics view. Ensure setup SQL was executed and data exists.", "error")
    df = pd.DataFrame(raw)
    names = ["accident_id", "accident_date", "state", "city", "location",
             "severity", "injuries", "fatalities", "company", "vehicle_type", "model_series"]
    if df.empty:
        df = pd.DataFrame(columns=names)
    f = {k: request.args.get(k, "").strip() for k in
         ("state", "city", "company", "vehicle_type", "model_series", "start", "end")}
    options = {
        "states": sorted(df.state.dropna().unique()),
        "cities": sorted(df[df.state.eq(f["state"])].city.dropna().unique()) if f["state"]
                 else sorted(df.city.dropna().unique()),
        "companies": sorted(df.company.dropna().unique()),
        "types": sorted(df.vehicle_type.dropna().unique()),
        "models": sorted(df.model_series.dropna().unique()),
    }
    # Reset a stale city filter when the state selector changes.
    if f["state"] and f["city"] and f["city"] not in options["cities"]:
        f["city"] = ""
    for field in ("state", "city", "company", "vehicle_type", "model_series"):
        if f[field]:
            df = df[df[field].eq(f[field])]
    if f["start"]:
        try:
            since = date.fromisoformat(f["start"])
            df = df[df.accident_date.ge(since.isoformat())]
        except ValueError:
            flash("Invalid From Date filter", "error")
    if f["end"]:
        try:
            until = date.fromisoformat(f["end"])
            df = df[df.accident_date.le(until.isoformat())]
        except ValueError:
            flash("Invalid To Date filter", "error")
    summary, charts, locations = report_payload(df)
    return render_template("reports.html", title="Reports", options=options, f=f,
                           summary=summary, charts=charts, locations=locations,
                           lookup=None, lookup_reg="", limited=limited, cap=10000)


@app.get("/reports/registration")
@login_required
def vehicle_lookup():
    reg = request.args.get("registration", "").strip().upper()
    if not re.fullmatch(r"[A-Z0-9 /-]{4,25}", reg):
        flash("Enter a valid registration or TR number.", "error")
        return redirect(url_for("reports"))
    try:
        found = private_lookup(reg)
    except Exception:
        flash("Vehicle lookup failed. Try again.", "error")
        return redirect(url_for("reports"))
    # Reuse the report page (includes full charts, then lookup results).
    raw, limited = analytics_rows()
    df = pd.DataFrame(raw)
    if df.empty:
        df = pd.DataFrame(columns=["accident_id", "state", "city", "company",
                                   "vehicle_type", "model_series", "accident_date",
                                   "fatalities", "severity"])
    summary, charts, locations = report_payload(df)
    options = {"states": sorted(df.state.dropna().unique()),
               "cities": sorted(df.city.dropna().unique()),
               "companies": sorted(df.company.dropna().unique()),
               "types": sorted(df.vehicle_type.dropna().unique()),
               "models": sorted(df.model_series.dropna().unique())}
    f = dict.fromkeys(("state", "city", "company", "vehicle_type", "model_series", "start", "end"), "")
    return render_template("reports.html", title="Reports", options=options, f=f,
                           summary=summary, charts=charts, locations=locations,
                           lookup=found, lookup_reg=reg, limited=limited, cap=10000)


@app.errorhandler(404)
def not_found(_):
    return "<h2>404 — Record not found</h2><a href='/'>Return to RADAS</a>", 404


def config_errors():
    """Check settings without ever printing passwords, API keys, or app secrets."""
    problems = []
    names = ("SUPABASE_URL", "SUPABASE_KEY", "FLASK_SECRET_KEY")
    for name in names:
        value = os.getenv(name, "").strip()
        if not value:
            problems.append(f"{name}: missing or empty")
        elif "YOUR_" in value.upper() or "REPLACE_ME" in value.upper():
            problems.append(f"{name}: still contains a placeholder")

    url = os.getenv("SUPABASE_URL", "").strip()
    if url and not any("SUPABASE_URL:" in x for x in problems):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            problems.append("SUPABASE_URL: must be a full http(s) URL")

    key = os.getenv("SUPABASE_KEY", "").strip()
    if key.startswith(("sb_secret_", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.service_role")):
        problems.append("SUPABASE_KEY: use the publishable/anon key, never a secret or service-role key")

    secret = os.getenv("FLASK_SECRET_KEY", "").strip()
    if secret and len(secret) < 32:
        problems.append("FLASK_SECRET_KEY: must contain at least 32 characters")
    return problems


def create_env_template():
    """Generate an editable .env only when one does not exist."""
    if DOTENV_PATH.exists():
        print(f"An .env file already exists: {DOTENV_PATH}")
        print("Edit it rather than creating another one.")
        return
    value = (
        "# Supabase Project Settings > API / API Keys\n"
        "SUPABASE_URL=https://YOUR_PROJECT.supabase.co\n"
        "SUPABASE_KEY=YOUR_PUBLISHABLE_KEY\n"
        f"FLASK_SECRET_KEY={secrets.token_hex(32)}\n"
    )
    DOTENV_PATH.write_text(value, encoding="utf-8")
    print(f"Created {DOTENV_PATH}")
    print("Fill in SUPABASE_URL and SUPABASE_KEY; keep FLASK_SECRET_KEY unchanged.")
    print("Do not commit .env to GitHub. Add .env to .gitignore.")


def print_config_diagnostics():
    print(f"Python file directory: {APP_DIR}")
    print(f"Expected .env location: {DOTENV_PATH}")
    print(f".env detected: {'YES' if DOTENV_PATH.is_file() else 'NO'}")
    for name in ("SUPABASE_URL", "SUPABASE_KEY", "FLASK_SECRET_KEY"):
        print(f"{name}: {'SET' if os.getenv(name, '').strip() else 'MISSING'}")
    issues = config_errors()
    if issues:
        print("\nConfiguration issues:")
        for issue in issues:
            print(f"- {issue}")
        print("\nFix: py RADAS_single_file_v1_3.py --init-env")
        print("Then edit the .env next to this Python file using your Supabase credentials.")
    else:
        print("\nConfiguration values look valid. This does not test network/database access.")
    return not issues


if __name__ == "__main__":
    if "--schema" in sys.argv:
        print(SQL)
        raise SystemExit(0)
    if "--init-env" in sys.argv:
        create_env_template()
        raise SystemExit(0)
    if "--check-env" in sys.argv:
        raise SystemExit(0 if print_config_diagnostics() else 1)
    if not print_config_diagnostics():
        raise SystemExit("\nStartup cancelled until the configuration is fixed.")
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
