# ml-pipelines/inference/server.py
#
# FastAPI inference server for Hearth Haven ML predictions.
# Loads trained .pkl pipelines and serves predictions via HTTP.
#
# USAGE (local development):
#   cd ml-pipelines/inference
#   pip install -r requirements.txt
#   uvicorn server:app --port 8000 --reload
#
# Reads AZURE_SQL_CONNECTIONSTRING from .env automatically.

import os
import pickle
import urllib.parse
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.pipeline import Pipeline as SklearnPipeline

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

MODELS_DIR        = Path(os.getenv("MODELS_DIR", Path(__file__).parent.parent / "models"))
CONNECTION_STRING = os.getenv("AZURE_SQL_CONNECTIONSTRING", "")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hearth-haven-ml")

app = FastAPI(title="Hearth Haven ML Inference", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5070",
        "http://localhost:5071",
        "http://localhost:5173",
        "https://hearth-haven.azurewebsites.net",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Database engine ────────────────────────────────────────────────────────────

def _get_engine() -> Engine:
    """Build a SQLAlchemy engine from the ADO.NET connection string in .env.

    The .env value uses ADO.NET key=value pairs (Encrypt=True, etc.).
    pyodbc needs different attribute names, so we translate them here.
    """
    # ADO.NET → pyodbc keyword mapping
    _adonet_to_pyodbc = {
        "server":                  "Server",
        "initial catalog":         "Database",
        "user id":                 "UID",
        "password":                "PWD",
        "encrypt":                 "Encrypt",          # True→yes, False→no below
        "trustservercertificate":  "TrustServerCertificate",
        "connection timeout":      "Connection Timeout",
        "multipleactiveresultsets": None,              # not a pyodbc attribute — skip
        "persist security info":   None,               # not a pyodbc attribute — skip
    }
    _bool_map = {"true": "yes", "false": "no"}

    pyodbc_parts = ["Driver={ODBC Driver 18 for SQL Server}"]
    for part in CONNECTION_STRING.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        key, _, val = part.partition("=")
        pyodbc_key = _adonet_to_pyodbc.get(key.strip().lower())
        if pyodbc_key is None:
            continue
        val = _bool_map.get(val.strip().lower(), val.strip())
        pyodbc_parts.append(f"{pyodbc_key}={val}")

    params = urllib.parse.quote_plus(";".join(pyodbc_parts))
    return create_engine(f"mssql+pyodbc:///?odbc_connect={params}")


# ── Model cache — loaded once, reused on every request ────────────────────────

_model_cache: dict[str, SklearnPipeline] = {}

def load_model(target_name: str) -> SklearnPipeline:
    if target_name not in _model_cache:
        path = MODELS_DIR / f"{target_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            model = pickle.load(f)
        assert isinstance(model, SklearnPipeline), \
            f"Expected sklearn Pipeline, got {type(model)}"
        _model_cache[target_name] = model
        log.info(f"Loaded model: {target_name}")
    return _model_cache[target_name]


# ── Feature builder — mirrors prepare_residents() for one resident ─────────────

def get_resident_features(resident_id: int) -> pd.DataFrame:
    """
    Query Azure SQL and assemble the feature vector for one resident.
    Mirrors every aggregation in prepare_residents() exactly.
    Returns a single-row DataFrame ready for pipeline.predict_proba().
    """
    try:
        engine = _get_engine()
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}")

    ref_date = pd.Timestamp(datetime.now(timezone.utc).date())

    with engine.connect() as con:

        # ── Resident base row ──────────────────────────────────────────────────
        r = pd.read_sql(text("""
            SELECT
                safehouse_id, case_status, birth_status, religion, case_category,
                referral_source, reintegration_type, reintegration_status,
                initial_risk_level, current_risk_level,
                sub_cat_orphaned, sub_cat_trafficked, sub_cat_child_labor,
                sub_cat_physical_abuse, sub_cat_sexual_abuse, sub_cat_osaec,
                sub_cat_cicl, sub_cat_at_risk, sub_cat_street_child,
                sub_cat_child_with_hiv, is_pwd, has_special_needs,
                pwd_type, special_needs_diagnosis,
                family_is_4ps, family_solo_parent, family_indigenous,
                family_parent_pwd, family_informal_settler,
                date_of_birth, date_of_admission, date_closed
            FROM dbo.residents
            WHERE resident_id = :rid
        """), con, params={"rid": resident_id})

        if r.empty:
            raise ValueError(f"Resident {resident_id} not found")

        row = r.iloc[0].copy()

        # Date-derived features
        dob      = pd.to_datetime(row["date_of_birth"],     errors="coerce")
        admitted = pd.to_datetime(row["date_of_admission"], errors="coerce")
        closed   = pd.to_datetime(row["date_closed"],       errors="coerce")

        row["age_at_admission_days"] = (admitted - dob).days \
            if pd.notna(dob) and pd.notna(admitted) else np.nan
        row["days_in_care"]          = (ref_date - admitted).days \
            if pd.notna(admitted) else np.nan
        row["length_of_stay_days"]   = (closed - admitted).days \
            if pd.notna(closed) and pd.notna(admitted) else row["days_in_care"]

        # Intentional null fills
        row["pwd_type"]                = row["pwd_type"]                or "None"
        row["special_needs_diagnosis"] = row["special_needs_diagnosis"] or "None"
        row["reintegration_type"]      = row["reintegration_type"]      or "None"
        row["reintegration_status"]    = row["reintegration_status"]    or "Not Started"

        # Risk numerics
        risk_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        row["initial_risk_num"] = risk_order.get(row["initial_risk_level"], np.nan)
        row["current_risk_num"] = risk_order.get(row["current_risk_level"], np.nan)

        # ── Health ─────────────────────────────────────────────────────────────
        h = pd.read_sql(text("""
            SELECT general_health_score, nutrition_score, sleep_quality_score,
                   energy_level_score, bmi, medical_checkup_done, record_date
            FROM dbo.health_wellbeing_records
            WHERE resident_id = :rid
            ORDER BY record_date DESC
        """), con, params={"rid": resident_id})

        if not h.empty:
            latest = h.iloc[0]
            row["health_score_latest"]    = latest["general_health_score"]
            row["nutrition_score_latest"] = latest["nutrition_score"]
            row["sleep_score_latest"]     = latest["sleep_quality_score"]
            row["energy_score_latest"]    = latest["energy_level_score"]
            row["bmi_latest"]             = latest["bmi"]
            row["checkup_completion"]     = h["medical_checkup_done"].mean()
        else:
            for col in ["health_score_latest", "nutrition_score_latest",
                        "sleep_score_latest", "energy_score_latest",
                        "bmi_latest", "checkup_completion"]:
                row[col] = np.nan

        # ── Education ──────────────────────────────────────────────────────────
        e = pd.read_sql(text("""
            SELECT attendance_rate, progress_percent, education_level,
                   enrollment_status, completion_status, record_date
            FROM dbo.education_records
            WHERE resident_id = :rid
            ORDER BY record_date DESC
        """), con, params={"rid": resident_id})

        if not e.empty:
            latest = e.iloc[0]
            row["attendance_rate_latest"]   = latest["attendance_rate"]
            row["progress_percent_latest"]  = latest["progress_percent"]
            row["education_level_latest"]   = latest["education_level"]
            row["enrollment_status_latest"] = latest["enrollment_status"]
            row["completion_status_latest"] = latest["completion_status"]
            row["attendance_avg"]           = e["attendance_rate"].mean()
            row["progress_avg"]             = e["progress_percent"].mean()
            row["education_records"]        = len(e)
        else:
            for col in ["attendance_rate_latest", "progress_percent_latest",
                        "attendance_avg", "progress_avg", "education_records"]:
                row[col] = np.nan
            for col in ["education_level_latest", "enrollment_status_latest",
                        "completion_status_latest"]:
                row[col] = None

        # ── Counseling ─────────────────────────────────────────────────────────
        c = pd.read_sql(text("""
            SELECT session_duration_minutes, progress_noted,
                   concerns_flagged, referral_made
            FROM dbo.process_recordings
            WHERE resident_id = :rid
        """), con, params={"rid": resident_id})

        if not c.empty:
            row["session_count"]        = len(c)
            row["avg_session_duration"] = c["session_duration_minutes"].mean()
            row["progress_rate"]        = c["progress_noted"].mean()
            row["concern_rate"]         = c["concerns_flagged"].mean()
            row["referral_rate"]        = c["referral_made"].mean()
        else:
            for col in ["session_count", "avg_session_duration", "progress_rate",
                        "concern_rate", "referral_rate"]:
                row[col] = 0

        # ── Incidents ──────────────────────────────────────────────────────────
        i = pd.read_sql(text("""
            SELECT incident_type, severity
            FROM dbo.incident_reports
            WHERE resident_id = :rid
        """), con, params={"rid": resident_id})

        severity_map = {"Low": 1, "Medium": 2, "High": 3}
        if not i.empty:
            i["severity_num"]          = i["severity"].map(severity_map)
            row["incident_count"]      = len(i)
            row["avg_severity"]        = i["severity_num"].mean()
            row["high_severity_count"] = int((i["severity_num"] == 3).sum())
            row["runaway_attempts"]    = int((i["incident_type"] == "RunawayAttempt").sum())
            row["self_harm_incidents"] = int((i["incident_type"] == "SelfHarm").sum())
        else:
            for col in ["incident_count", "high_severity_count",
                        "runaway_attempts", "self_harm_incidents"]:
                row[col] = 0
            row["avg_severity"] = np.nan

        # ── Visitations ────────────────────────────────────────────────────────
        v = pd.read_sql(text("""
            SELECT safety_concerns_noted, visit_outcome, family_cooperation_level
            FROM dbo.home_visitations
            WHERE resident_id = :rid
        """), con, params={"rid": resident_id})

        coop_map = {"Highly Cooperative": 4, "Cooperative": 3,
                    "Neutral": 2, "Uncooperative": 1}
        if not v.empty:
            v["coop_num"]                   = v["family_cooperation_level"].map(coop_map)
            row["visitation_count"]         = len(v)
            row["safety_concern_rate"]      = v["safety_concerns_noted"].mean()
            row["favorable_outcome_rate"]   = (v["visit_outcome"] == "Favorable").sum() / len(v)
            row["family_cooperation_score"] = v["coop_num"].mean()
        else:
            for col in ["visitation_count", "safety_concern_rate",
                        "favorable_outcome_rate", "family_cooperation_score"]:
                row[col] = 0

        # ── Interventions ──────────────────────────────────────────────────────
        ip = pd.read_sql(text("""
            SELECT plan_id, status, plan_category
            FROM dbo.intervention_plans
            WHERE resident_id = :rid
        """), con, params={"rid": resident_id})

        if not ip.empty:
            row["total_plans"]           = len(ip)
            row["open_plans"]            = int((ip["status"] == "Open").sum())
            row["plan_categories"]       = ip["plan_category"].nunique()
            row["plan_achievement_rate"] = (ip["status"] == "Achieved").sum() / len(ip)
        else:
            for col in ["total_plans", "open_plans", "plan_categories"]:
                row[col] = 0
            row["plan_achievement_rate"] = 0.0

    # Drop raw date columns — not used by the model
    row = row.drop(labels=["date_of_birth", "date_of_admission", "date_closed"],
                   errors="ignore")

    return pd.DataFrame([row])


# ── Response models ────────────────────────────────────────────────────────────

class ReintegrationResponse(BaseModel):
    resident_id:     int
    readiness_score: float
    probability:     float
    recommendation:  str
    model_version:   str
    predicted_at:    str

class HealthCheckResponse(BaseModel):
    status:        str
    models_loaded: list[str]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    return {
        "status":        "ok",
        "models_loaded": list(_model_cache.keys()),
    }


@app.get("/predict/reintegration/{resident_id}",
         response_model=ReintegrationResponse)
def predict_reintegration(resident_id: int):
    """
    Returns a Reintegration Readiness Score (0–100) for one resident.
    Called by the .NET backend — not directly by the React frontend.
    """
    try:
        pipeline = load_model("reintegration_achieved")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        features = get_resident_features(resident_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        proba = float(pipeline.predict_proba(features)[0, 1])
    except Exception as e:
        log.error(f"Prediction failed for resident {resident_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    score = round(proba * 100, 1)

    def recommend(s: float) -> str:
        if s >= 75: return "Reintegration readiness indicated — schedule case conference"
        if s >= 50: return "Progressing — continue current intervention plan"
        return "Continue structured care — reintegration readiness not yet indicated"

    return {
        "resident_id":     resident_id,
        "readiness_score": score,
        "probability":     round(proba, 4),
        "recommendation":  recommend(score),
        "model_version":   "reintegration_achieved_v1",
        "predicted_at":    datetime.now(timezone.utc).isoformat(),
    }