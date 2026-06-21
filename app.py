"""
MediSense AI backend.
Flask + SQLite + JWT auth + Groq proxy + Edge-TTS (multi-language read-aloud).
Run locally:  python app.py
Deploy:       gunicorn app:app   (see README.md)
"""
import os
import datetime
import functools
import asyncio

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-this-in-render-env-vars")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///medisense.db")
# Render's Postgres URLs start with postgres://, SQLAlchemy wants postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app)  # allow the Netlify frontend to call this API
db = SQLAlchemy(app)

TOKEN_EXP_DAYS = 30

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(160), unique=True, nullable=False)
    phone = db.Column(db.String(30), default="")
    blood = db.Column(db.String(10), default="")
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "name": self.full_name, "email": self.email,
            "phone": self.phone, "blood": self.blood,
        }


class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    medicine_name = db.Column(db.String(120), nullable=False)
    dosage = db.Column(db.String(60), default="")
    reminder_time = db.Column(db.String(20), default="")
    frequency = db.Column(db.String(60), default="Once daily")
    done = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "name": self.medicine_name, "dose": self.dosage or "—",
            "time": self.reminder_time, "freq": self.frequency, "done": self.done,
        }


class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    doctor = db.Column(db.String(120), nullable=False)
    speciality = db.Column(db.String(120), default="")
    appt_date = db.Column(db.String(20), nullable=False)
    appt_time = db.Column(db.String(20), default="")
    reason = db.Column(db.String(255), default="")
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id, "doctor": self.doctor, "spec": self.speciality,
            "date": self.appt_date, "time": self.appt_time, "reason": self.reason or "",
        }


with app.app_context():
    db.create_all()

# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def make_token(user_id):
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=TOKEN_EXP_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401
        token = auth.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        user = User.query.get(payload["user_id"])
        if not user:
            return jsonify({"error": "User not found"}), 401
        request.user = user
        return f(*args, **kwargs)
    return wrapper

# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True, silent=True) or {}
    fname = (data.get("fname") or "").strip()
    lname = (data.get("lname") or "").strip()
    email = (data.get("email") or "").strip().lower()
    phone = (data.get("phone") or "").strip()
    password = data.get("password") or ""
    blood = (data.get("blood") or "").strip()

    if not fname or not email or not password:
        return jsonify({"error": "Name, email and password are required"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "An account with this email already exists"}), 409

    full_name = f"{fname} {lname}".strip()
    user = User(
        full_name=full_name, email=email, phone=phone, blood=blood,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()
    token = make_token(user.id)
    return jsonify({"token": token, "user": user.to_dict()}), 201


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True, silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid email or password"}), 401
    token = make_token(user.id)
    return jsonify({"token": token, "user": user.to_dict()})


@app.route("/api/me", methods=["GET"])
@login_required
def me():
    return jsonify({"user": request.user.to_dict()})

# ---------------------------------------------------------------------------
# Reminders
# ---------------------------------------------------------------------------
@app.route("/api/reminders", methods=["GET"])
@login_required
def list_reminders():
    rows = Reminder.query.filter_by(user_id=request.user.id).order_by(Reminder.created_at).all()
    return jsonify({"reminders": [r.to_dict() for r in rows]})


@app.route("/api/reminders", methods=["POST"])
@login_required
def add_reminder():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Medicine name is required"}), 400
    r = Reminder(
        user_id=request.user.id, medicine_name=name,
        dosage=(data.get("dose") or "").strip(),
        reminder_time=(data.get("time") or "").strip(),
        frequency=(data.get("freq") or "Once daily").strip(),
        done=False,
    )
    db.session.add(r)
    db.session.commit()
    return jsonify({"reminder": r.to_dict()}), 201


@app.route("/api/reminders/<int:rid>", methods=["PATCH"])
@login_required
def update_reminder(rid):
    r = Reminder.query.filter_by(id=rid, user_id=request.user.id).first()
    if not r:
        return jsonify({"error": "Not found"}), 404
    data = request.get_json(force=True, silent=True) or {}
    if "done" in data:
        r.done = bool(data["done"])
    db.session.commit()
    return jsonify({"reminder": r.to_dict()})


@app.route("/api/reminders/<int:rid>", methods=["DELETE"])
@login_required
def delete_reminder(rid):
    r = Reminder.query.filter_by(id=rid, user_id=request.user.id).first()
    if not r:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(r)
    db.session.commit()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# Appointments
# ---------------------------------------------------------------------------
@app.route("/api/appointments", methods=["GET"])
@login_required
def list_appts():
    rows = Appointment.query.filter_by(user_id=request.user.id).order_by(Appointment.created_at).all()
    return jsonify({"appointments": [a.to_dict() for a in rows]})


@app.route("/api/appointments", methods=["POST"])
@login_required
def add_appt():
    data = request.get_json(force=True, silent=True) or {}
    doctor = (data.get("doctor") or "").strip()
    date = (data.get("date") or "").strip()
    if not doctor or not date:
        return jsonify({"error": "Doctor and date are required"}), 400
    a = Appointment(
        user_id=request.user.id, doctor=doctor,
        speciality=(data.get("spec") or "").strip(),
        appt_date=date, appt_time=(data.get("time") or "").strip(),
        reason=(data.get("reason") or "").strip(),
    )
    db.session.add(a)
    db.session.commit()
    return jsonify({"appointment": a.to_dict()}), 201


@app.route("/api/appointments/<int:aid>", methods=["DELETE"])
@login_required
def delete_appt(aid):
    a = Appointment.query.filter_by(id=aid, user_id=request.user.id).first()
    if not a:
        return jsonify({"error": "Not found"}), 404
    db.session.delete(a)
    db.session.commit()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# AI chat + symptom checker (Groq proxy — key never reaches the browser)
# ---------------------------------------------------------------------------
import urllib.request
import json as _json

def _call_groq(messages, max_tokens=500, temperature=0.7):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured on the server")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=_json.dumps({
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode())
    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Groq error"))
    return data["choices"][0]["message"]["content"]


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history") or []
    lang_name = (data.get("langName") or "English").strip()

    sys_msg = (
        f"You are MediSense AI, a helpful health assistant. The user's preferred "
        f"language is {lang_name}. If {lang_name} is not English, reply in "
        f"{lang_name}. Answer health questions clearly in plain sentences under "
        f"100 words. No bullet points or markdown. Always recommend consulting a "
        f"doctor for serious concerns. Be warm and practical."
    )
    messages = [{"role": "system", "content": sys_msg}] + [
        {"role": m.get("role"), "content": m.get("content")} for m in history
    ]
    try:
        reply = _call_groq(messages, max_tokens=500, temperature=0.7)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/analyze-symptoms", methods=["POST"])
def analyze_symptoms():
    data = request.get_json(force=True, silent=True) or {}
    symptoms = data.get("symptoms") or []
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    sys_p = (
        'You are a medical AI. Given symptoms respond ONLY with valid JSON like '
        'this exact format: {"conditions":[{"name":"Flu","probability":82,'
        '"severity":"moderate"},{"name":"Cold","probability":60,"severity":"low"}],'
        '"precaution":"Rest and drink fluids.","urgency":"routine"} Rules: severity '
        'must be low/moderate/high, urgency must be routine/soon/emergency, give '
        '2-4 conditions, output ONLY the JSON object nothing else no markdown no backticks.'
    )
    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": "Symptoms: " + ", ".join(symptoms)},
    ]
    try:
        raw = _call_groq(messages, max_tokens=1000, temperature=0.7)
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        result = _json.loads(cleaned)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 502

# ---------------------------------------------------------------------------
# Text-to-speech (Edge TTS) — fixes "read aloud" for non-English languages
# ---------------------------------------------------------------------------
import edge_tts

# One reliable neural voice per language — these all genuinely speak the
# language natively (unlike the old browser/ResponsiveVoice fallback chain).
VOICE_MAP = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-SapnaNeural",
    "te": "te-IN-ShrutiNeural",
    "ta": "ta-IN-PallaviNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ml": "ml-IN-SobhanaNeural",
}


@app.route("/api/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    lang = (data.get("lang") or "en").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if len(text) > 2000:
        text = text[:2000]

    voice = VOICE_MAP.get(lang, VOICE_MAP["en"])
    out_path = f"/tmp/tts_{os.getpid()}_{abs(hash(text)) % 100000}.mp3"

    async def _gen():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out_path)

    try:
        asyncio.run(_gen())
        return send_file(out_path, mimetype="audio/mpeg")
    except Exception as e:
        return jsonify({"error": f"TTS failed: {e}"}), 502
    finally:
        try:
            if os.path.exists(out_path):
                # Clean up after response is sent (best effort, next request will overwrite anyway)
                pass
        except Exception:
            pass


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
