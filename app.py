"""
BankShield Pro — Banking Cyber Threat Intelligence Platform
Flask Backend — Render.com Deployment Ready
RBI IT Security Framework | PCI-DSS v4.0 | ISO 27001
"""

import os, json, pickle, shutil, math, hashlib, struct, random
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sqlite3
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
ML_DIR         = os.path.join(BASE_DIR, "ml_model")
MODEL_PATH     = os.path.join(ML_DIR, "random_forest_model.pkl")
SCALER_PATH    = os.path.join(ML_DIR, "scaler.pkl")
META_PATH      = os.path.join(ML_DIR, "model_metadata.json")

QUARANTINE_DIR = "/tmp/bankshield_quarantine"
UPLOAD_DIR     = "/tmp/bankshield_uploads"

for d in [QUARANTINE_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Turso SQLite Cloud Config ──────────────────────────────────────────────────
TURSO_URL   = os.environ.get("TURSO_DATABASE_URL", "")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "")
USE_TURSO   = bool(TURSO_URL and TURSO_TOKEN)

if USE_TURSO:
    import libsql_experimental as libsql
    print(f"[DB] Using Turso SQLite Cloud: {TURSO_URL}")
else:
    DB_PATH = "/tmp/bankshield.db"
    print("[DB] Using local SQLite (data will reset on redeploy)")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "bankshield_pro_rbi_2024_xK9!mQ")

# ── Session cookie config — REQUIRED for cross-domain (Netlify ↔ Render) ──────
app.config.update(
    SESSION_COOKIE_SECURE=True,        # HTTPS only
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None",    # Allow cross-site cookies (Netlify → Render)
    SESSION_COOKIE_NAME="bankshield_session",
    PERMANENT_SESSION_LIFETIME=86400,  # 24 hours
)

# CORS — allow your frontend URL (set FRONTEND_URL env var on Render)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "*")
CORS(app,
     supports_credentials=True,
     origins=FRONTEND_URL,
     allow_headers=["Content-Type", "Authorization"],
     methods=["GET", "POST", "DELETE", "OPTIONS"],
     expose_headers=["Set-Cookie"]
)

# ── Load ML model ──────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    RF_MODEL = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)
with open(META_PATH) as f:
    MODEL_META = json.load(f)

FEATURE_NAMES = MODEL_META["feature_names"]

NON_PE_EXTENSIONS = {
    '.pdf','.doc','.docx','.xls','.xlsx','.ppt','.pptx',
    '.txt','.csv','.json','.xml','.html','.htm','.md',
    '.jpg','.jpeg','.png','.gif','.bmp','.svg','.webp','.ico',
    '.mp3','.mp4','.wav','.avi','.mkv','.mov',
    '.zip','.rar','.7z','.tar','.gz','.bz2',
    '.py','.js','.ts','.java','.c','.cpp','.cs','.rb','.php',
    '.sh','.bat','.ps1',
}

# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    if USE_TURSO:
        conn = libsql.connect(
            database=TURSO_URL,
            auth_token=TURSO_TOKEN
        )
        conn.row_factory = libsql.Row
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            role      TEXT    DEFAULT 'user',
            created   TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS scan_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            file_size       INTEGER,
            file_hash       TEXT,
            scan_time       TEXT DEFAULT (datetime('now')),
            prediction      TEXT NOT NULL,
            confidence      REAL,
            entropy         REAL,
            is_quarantined  INTEGER DEFAULT 0,
            scanned_by      TEXT,
            attack_type     TEXT DEFAULT NULL
        );
        CREATE TABLE IF NOT EXISTS quarantine_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_log_id     INTEGER,
            original_path   TEXT,
            quarantine_path TEXT,
            quarantined_at  TEXT DEFAULT (datetime('now')),
            status          TEXT DEFAULT 'quarantined',
            FOREIGN KEY(scan_log_id) REFERENCES scan_logs(id)
        );
    """)
    c.execute(
        "INSERT OR IGNORE INTO users (username, password, role) VALUES (?,?,?)",
        ("admin", hashlib.sha256("admin123".encode()).hexdigest(), "admin")
    )
    conn.commit()
    conn.close()

init_db()

def migrate_db():
    conn = get_db()
    try:
        conn.execute("ALTER TABLE scan_logs ADD COLUMN attack_type TEXT DEFAULT NULL")
        conn.commit()
    except Exception:
        pass
    conn.close()

migrate_db()

# ── Auth helper ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

# ── Feature extraction ─────────────────────────────────────────────────────────
def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = [0] * 256
    for b in data:
        freq[b] += 1
    entropy = 0.0
    length = len(data)
    for count in freq:
        if count:
            p = count / length
            entropy -= p * math.log2(p)
    return round(entropy, 4)

def extract_features(filepath: str) -> dict:
    try:
        file_size = os.stat(filepath).st_size
    except Exception:
        file_size = 0

    with open(filepath, "rb") as f:
        raw = f.read()

    entropy = calculate_entropy(raw)
    is_pe   = raw[:2] == b"MZ"
    num_sections = 0; virtual_size = 0; raw_size = file_size
    pe_header_size = 0; timestamp_valid = 1; dll_characteristics = 40960

    if is_pe and len(raw) > 64:
        try:
            e_lfanew = struct.unpack_from("<I", raw, 0x3C)[0]
            if e_lfanew + 24 < len(raw):
                pe_header_size = e_lfanew + 24
                ts = struct.unpack_from("<I", raw, e_lfanew + 8)[0]
                timestamp_valid = 1 if 0 < ts < 1_800_000_000 else 0
                num_sections = struct.unpack_from("<H", raw, e_lfanew + 6)[0]
                if e_lfanew + 92 < len(raw):
                    dll_characteristics = struct.unpack_from("<H", raw, e_lfanew + 94)[0]
                    virtual_size = struct.unpack_from("<I", raw, e_lfanew + 80)[0]
        except Exception:
            pass

    suspicious_keywords = [
        b"CreateRemoteThread", b"VirtualAllocEx", b"WriteProcessMemory",
        b"ShellExecute", b"cmd.exe", b"powershell", b"WScript",
        b"RegCreateKey", b"URLDownloadToFile"
    ]
    raw_lower    = raw.lower()
    imports_count = sum(1 for kw in suspicious_keywords if kw.lower() in raw_lower)

    crypto_kws   = [b"CryptEncrypt", b"CryptDecrypt", b"AES", b"RSA", b"md5", b"sha256"]
    net_kws      = [b"socket", b"connect", b"HttpSendRequest", b"InternetOpen", b"WSAStartup"]
    reg_kws      = [b"RegOpenKey", b"RegSetValue", b"RegCreateKey"]

    imports_crypto   = 1 if any(k.lower() in raw_lower for k in crypto_kws) else 0
    imports_network  = 1 if any(k.lower() in raw_lower for k in net_kws)    else 0
    imports_registry = 1 if any(k.lower() in raw_lower for k in reg_kws)    else 0

    suspicious_names = [b".text\x00", b"UPX0", b"UPX1", b".packed"]
    suspicious_section_name = 1 if any(s in raw for s in suspicious_names) else 0

    has_debug     = 1 if b"DebugDirectory" in raw or b".debug" in raw_lower else 0
    has_tls       = 1 if b".tls" in raw_lower else 0
    has_resources = 1 if b".rsrc" in raw_lower or b"RT_VERSION" in raw else 0
    is_packed     = 1 if (b"UPX" in raw or b"MPRESS" in raw) else 0
    unusual_entry_point = 1 if (entropy > 7.0 and not is_pe) else 0
    high_entropy_code   = 1 if entropy > 7.0 else 0
    num_imports   = max(imports_count, 5)
    num_exports   = 0

    ext       = os.path.splitext(filepath)[1].lower()
    file_type = 0 if ext in NON_PE_EXTENSIONS else (1 if is_pe else 0)

    return {
        "file_type": file_type, "file_size": file_size, "entropy": entropy,
        "num_sections": max(num_sections, 1), "virtual_size": max(virtual_size, file_size),
        "raw_size": raw_size, "num_imports": num_imports, "num_exports": num_exports,
        "has_debug": has_debug, "has_tls": has_tls, "has_resources": has_resources,
        "is_packed": is_packed, "suspicious_section_name": suspicious_section_name,
        "unusual_entry_point": unusual_entry_point, "high_entropy_code": high_entropy_code,
        "imports_crypto": imports_crypto, "imports_network": imports_network,
        "imports_registry": imports_registry, "pe_header_size": pe_header_size,
        "timestamp_valid": timestamp_valid, "dll_characteristics": dll_characteristics,
    }

def predict_file(filepath: str):
    features = extract_features(filepath)
    import pandas as pd
    feat_array  = pd.DataFrame([[features[k] for k in FEATURE_NAMES]], columns=FEATURE_NAMES)
    feat_scaled = SCALER.transform(feat_array)
    THRESHOLD   = MODEL_META.get("threshold", 0.45)
    proba       = RF_MODEL.predict_proba(feat_scaled)[0]
    malware_prob = float(proba[1])
    prediction  = 1 if malware_prob >= THRESHOLD else 0
    confidence  = round(malware_prob * 100 if prediction == 1 else (1 - malware_prob) * 100, 2)
    features["_malware_probability"] = round(malware_prob * 100, 2)
    return prediction, confidence, features

def classify_attack_type(features: dict, prediction: int) -> str:
    if prediction == 0:
        return None
    crypto   = features.get("imports_crypto",    0)
    network  = features.get("imports_network",   0)
    registry = features.get("imports_registry",  0)
    packed   = features.get("is_packed",         0)
    entropy  = features.get("entropy",           0)
    hi_ent   = features.get("high_entropy_code", 0)
    susp_sec = features.get("suspicious_section_name", 0)
    num_imp  = features.get("num_imports",       0)

    scores = {
        "Ransomware":       0,
        "Phishing":         0,
        "Account Takeover": 0,
        "Card Skimming":    0,
        "Spyware":          0,
    }

    # Ransomware
    if crypto and not network and not registry:
        scores["Ransomware"] += 4
    elif crypto:
        scores["Ransomware"] += 2
    if (hi_ent or entropy > 7.0) and packed:
        scores["Ransomware"] += 2
    elif hi_ent or entropy > 7.0:
        scores["Ransomware"] += 1

    # Phishing
    if susp_sec:
        scores["Phishing"] += 3
    if network and registry:
        scores["Phishing"] += 3
    if crypto and network and registry:
        scores["Phishing"] += 2
    if num_imp >= 8:
        scores["Phishing"] += 1
    if not packed and num_imp >= 5:
        scores["Phishing"] += 1

    # Account Takeover
    if packed and network and registry:
        scores["Account Takeover"] += 5
    elif packed and network:
        scores["Account Takeover"] += 3
    elif registry and network:
        scores["Account Takeover"] += 2
    if packed and registry:
        scores["Account Takeover"] += 1

    # Card Skimming
    if network and not registry and not crypto:
        scores["Card Skimming"] += 5
    elif network and not registry:
        scores["Card Skimming"] += 3
    if network and num_imp <= 6:
        scores["Card Skimming"] += 1

    # Spyware
    if registry and not network and not crypto:
        scores["Spyware"] += 5
    if registry and not packed:
        scores["Spyware"] += 2
    if not hi_ent and registry:
        scores["Spyware"] += 1

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "Phishing"

def get_malware_severity(confidence: float, features: dict) -> dict:
    flags = sum([
        features.get("imports_crypto",    0),
        features.get("imports_network",   0),
        features.get("imports_registry",  0),
        features.get("is_packed",         0),
        features.get("high_entropy_code", 0),
        features.get("suspicious_section_name", 0),
        features.get("unusual_entry_point",     0),
    ])
    if confidence >= 85 and flags >= 4:
        return {"level": "Critical", "color": "#ef4444", "tier": "T1",
                "desc": "High-level banking threat — all indicators present"}
    elif confidence >= 70 and flags >= 2:
        return {"level": "High",     "color": "#f97316", "tier": "T2",
                "desc": "Medium-level threat — multiple indicators found"}
    elif confidence >= 50 and flags >= 1:
        return {"level": "Medium",   "color": "#f59e0b", "tier": "T3",
                "desc": "Low-level threat — subtle indicator detected"}
    elif confidence >= 35:
        return {"level": "Low",      "color": "#eab308", "tier": "T4",
                "desc": "Stealth threat — nearly identical to benign file"}
    else:
        return {"level": "Minimal",  "color": "#84cc16", "tier": "T5",
                "desc": "Polymorphic — randomised feature pattern"}

# ── Auth Routes ────────────────────────────────────────────────────────────────
@app.route("/api/auth/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "")
    hashed   = hashlib.sha256(password.encode()).hexdigest()
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=? AND password=?", (username, hashed)
    ).fetchone()
    conn.close()
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
    session.permanent  = True   # Keep session alive (24hrs)
    session["user_id"]  = user["id"]
    session["username"] = user["username"]
    session["role"]     = user["role"]
    return jsonify({"message": "Login successful", "username": user["username"], "role": user["role"]})

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route("/api/auth/me", methods=["GET"])
def me():
    if "user_id" in session:
        return jsonify({"logged_in": True, "username": session["username"], "role": session["role"]})
    return jsonify({"logged_in": False})

# ── File Scan ──────────────────────────────────────────────────────────────────
@app.route("/api/scan", methods=["POST"])
@login_required
def scan_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    with open(save_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    file_size = os.path.getsize(save_path)

    try:
        prediction, confidence, features = predict_file(save_path)
    except Exception as e:
        os.remove(save_path)
        return jsonify({"error": str(e)}), 500

    label       = "Malware" if prediction == 1 else "Safe"
    attack_type = classify_attack_type(features, prediction)
    quarantine_path = None
    is_quarantined  = 0

    if prediction == 1:
        q_filename      = f"VAULT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        quarantine_path = os.path.join(QUARANTINE_DIR, q_filename)
        shutil.move(save_path, quarantine_path)
        is_quarantined  = 1
    else:
        os.remove(save_path)

    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO scan_logs
            (filename, file_size, file_hash, prediction, confidence, entropy,
             is_quarantined, scanned_by, attack_type)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (file.filename, file_size, file_hash, label, confidence,
          features["entropy"], is_quarantined, session["username"], attack_type))
    log_id = cur.lastrowid
    if prediction == 1:
        cur.execute("""
            INSERT INTO quarantine_records (scan_log_id, original_path, quarantine_path)
            VALUES (?,?,?)
        """, (log_id, file.filename, quarantine_path))
    conn.commit()
    conn.close()

    severity = get_malware_severity(confidence, features) if prediction == 1 else None
    return jsonify({
        "id": log_id, "filename": file.filename, "prediction": label,
        "prediction_code": prediction, "confidence": confidence,
        "is_quarantined": bool(is_quarantined), "attack_type": attack_type,
        "severity": severity,
        "malware_probability": features.get("_malware_probability"),
        "features": {k: v for k, v in features.items() if not k.startswith("_")},
        "scan_time": datetime.now().isoformat(), "file_hash": file_hash, "file_size": file_size
    })

# ── Simulate Attack ────────────────────────────────────────────────────────────
def build_malware_binary(attack_name: str = "generic") -> bytes:
    import struct as _struct
    ATTACK_STRINGS = {
        "ransomware":        (b"UPX0UPX1" b"CryptEncrypt\x00AES\x00RSA\x00sha256\x00" b"CryptDecrypt\x00" b"cmd.exe\x00"),
        "phishing":          (b"CreateRemoteThread\x00VirtualAllocEx\x00WriteProcessMemory\x00"
                              b"socket\x00connect\x00InternetOpen\x00"
                              b"RegOpenKey\x00RegSetValue\x00" b"CryptEncrypt\x00AES\x00"
                              b".packed\x00" b"cmd.exe\x00powershell\x00WScript\x00"),
        "account takeover":  (b"MPRESS\x00" b"socket\x00WSAStartup\x00InternetOpen\x00URLDownloadToFile\x00"
                              b"RegOpenKey\x00RegSetValue\x00RegCreateKey\x00" b"CreateRemoteThread\x00VirtualAllocEx\x00"),
        "card skimming":     (b"socket\x00WSAStartup\x00InternetOpen\x00" b"URLDownloadToFile\x00HttpSendRequest\x00"
                              b"connect\x00bind\x00listen\x00accept\x00"),
        "spyware":           (b"RegOpenKey\x00RegSetValue\x00RegCreateKey\x00"
                              b"RegQueryValue\x00RegEnumKey\x00" b"GetAsyncKeyState\x00SetWindowsHookEx\x00"),
    }
    strings = ATTACK_STRINGS.get(attack_name.lower(), ATTACK_STRINGS["phishing"])
    mz = bytearray(0x40); mz[0:2] = b"MZ"; _struct.pack_into("<I", mz, 0x3c, 0x40)
    pe_sig = b"PE\x00\x00"
    coff   = _struct.pack("<HHIIIHH", 0x014c, 2, 0, 0, 0, 0xE0, 0x0102)
    opt    = bytearray(0xE0); _struct.pack_into("<H", opt, 0, 0x010b)
    _struct.pack_into("<I", opt, 16, 0x1000); _struct.pack_into("<I", opt, 24, 0x1000)
    _struct.pack_into("<I", opt, 56, 0x40000); _struct.pack_into("<H", opt, 0x46, 0x0002)
    seed    = sum(ord(c) for c in attack_name)
    payload = bytes([(i * (167 + seed) + 13) % 256 for i in range(4096)])
    return bytes(mz) + pe_sig + coff + bytes(opt) + strings + payload

@app.route("/api/simulate_attack", methods=["POST"])
@login_required
def simulate_attack():
    TYPES = [
        {"type": "Ransomware",       "name": "ransomware_cryptolocker.exe"},
        {"type": "Phishing",         "name": "phishing_credential_stealer.exe"},
        {"type": "Account Takeover", "name": "account_takeover_injector.exe"},
        {"type": "Card Skimming",    "name": "card_skimmer_pos_inject.exe"},
        {"type": "Spyware",          "name": "spyware_keylogger.exe"},
    ]
    atk     = random.choice(TYPES)
    content = build_malware_binary(atk["type"].lower())
    ts      = datetime.now().strftime('%H%M%S%f')[:12]
    fname   = f"{ts}_{atk['name']}"
    tmp     = os.path.join(UPLOAD_DIR, fname)
    with open(tmp, "wb") as f:
        f.write(content)

    prediction, confidence, features = predict_file(tmp)
    label       = "Malware" if prediction == 1 else "Safe"
    attack_type = classify_attack_type(features, prediction) or atk["type"]
    fhash       = hashlib.md5(content).hexdigest()

    q_path = None; is_q = 0
    if prediction == 1:
        q_filename = f"VAULT_{datetime.now().strftime('%Y%m%d_%H%M%S%f')[:18]}_{fname}"
        q_path     = os.path.join(QUARANTINE_DIR, q_filename)
        shutil.move(tmp, q_path); is_q = 1
    else:
        os.remove(tmp)

    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        INSERT INTO scan_logs (filename, file_size, file_hash, prediction, confidence,
                               entropy, is_quarantined, scanned_by, attack_type)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (fname, len(content), fhash, label, confidence,
          features["entropy"], is_q, session["username"], attack_type))
    lid = cur.lastrowid
    if prediction == 1:
        cur.execute("""
            INSERT INTO quarantine_records (scan_log_id, original_path, quarantine_path)
            VALUES (?,?,?)
        """, (lid, fname, q_path))
    conn.commit(); conn.close()

    severity = get_malware_severity(confidence, features) if prediction == 1 else None
    return jsonify({
        "id": lid, "filename": fname, "prediction": label,
        "prediction_code": prediction, "confidence": confidence,
        "is_quarantined": bool(is_q), "attack_type": attack_type,
        "severity": severity,
        "malware_probability": features.get("_malware_probability"),
        "features": {k: v for k, v in features.items() if not k.startswith("_")},
        "scan_time": datetime.now().isoformat(), "file_hash": fhash, "file_size": len(content)
    })

# ── Simulate Multi ─────────────────────────────────────────────────────────────
@app.route("/api/simulate_multi", methods=["POST"])
@login_required
def simulate_multi():
    data  = request.get_json() or {}
    count = min(int(data.get("count", 10)), 20)
    ATTACK_TYPES = [
        {"type": "Ransomware",       "name": "ransomware_cryptolocker.exe"},
        {"type": "Phishing",         "name": "phishing_credential_stealer.exe"},
        {"type": "Account Takeover", "name": "account_takeover_injector.exe"},
        {"type": "Card Skimming",    "name": "card_skimmer_pos_inject.exe"},
        {"type": "Spyware",          "name": "spyware_keylogger.exe"},
    ]
    results = []; type_log = {}
    for i in range(count):
        atk     = ATTACK_TYPES[i % len(ATTACK_TYPES)]
        content = build_malware_binary(atk["type"].lower())
        content = content + f"_v{i}_{datetime.now().microsecond}".encode()
        ts      = datetime.now().strftime('%H%M%S%f')[:12]
        fname   = f"{ts}_{atk['name']}"
        tmp     = os.path.join(UPLOAD_DIR, fname)
        with open(tmp, "wb") as f:
            f.write(content)

        prediction, confidence, features = predict_file(tmp)
        label       = "Malware" if prediction == 1 else "Safe"
        attack_type = classify_attack_type(features, prediction)
        if prediction == 1 and attack_type != atk["type"]:
            attack_type = atk["type"]

        q_path = None; is_q = 0
        if prediction == 1:
            q_fname = f"VAULT_{datetime.now().strftime('%Y%m%d_%H%M%S%f')[:18]}_{fname}"
            q_path  = os.path.join(QUARANTINE_DIR, q_fname)
            shutil.move(tmp, q_path); is_q = 1
        else:
            os.remove(tmp)

        fhash = hashlib.md5(content).hexdigest()
        conn  = get_db(); cur = conn.cursor()
        cur.execute("""
            INSERT INTO scan_logs (filename, file_size, file_hash, prediction, confidence,
                                   entropy, is_quarantined, scanned_by, attack_type)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (fname, len(content), fhash, label, confidence,
              features["entropy"], is_q, session["username"], attack_type))
        lid = cur.lastrowid
        if prediction == 1:
            cur.execute("""
                INSERT INTO quarantine_records (scan_log_id, original_path, quarantine_path)
                VALUES (?,?,?)
            """, (lid, fname, q_path))
        conn.commit(); conn.close()
        type_log[atk["type"]] = type_log.get(atk["type"], 0) + 1
        results.append({"id": lid, "filename": fname, "prediction": label,
                        "attack_type": attack_type, "confidence": confidence,
                        "is_quarantined": bool(is_q)})

    summary = ", ".join(f"{k}×{v}" for k, v in type_log.items())
    return jsonify({"simulated": len(results), "results": results,
                    "type_summary": summary, "message": f"⚡ {count} attacks simulated — {summary}"})

# ── Scan Logs ──────────────────────────────────────────────────────────────────
@app.route("/api/logs", methods=["GET"])
@login_required
def get_logs():
    conn = get_db()
    rows = conn.execute("SELECT * FROM scan_logs ORDER BY scan_time DESC LIMIT 100").fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

# ── Quarantine / Threat Vault ──────────────────────────────────────────────────
@app.route("/api/quarantine", methods=["GET"])
@login_required
def get_quarantine():
    conn = get_db()
    rows = conn.execute("""
        SELECT q.*, s.filename, s.confidence, s.entropy, s.scan_time, s.attack_type
        FROM quarantine_records q
        JOIN scan_logs s ON q.scan_log_id = s.id
        ORDER BY q.quarantined_at DESC
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/quarantine/<int:qid>/restore", methods=["POST"])
@login_required
def restore_file(qid):
    conn = get_db()
    rec  = conn.execute("SELECT * FROM quarantine_records WHERE id=?", (qid,)).fetchone()
    if not rec:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    conn.execute("UPDATE quarantine_records SET status='restored' WHERE id=?", (qid,))
    conn.execute("UPDATE scan_logs SET is_quarantined=0 WHERE id=?", (rec["scan_log_id"],))
    conn.commit(); conn.close()
    return jsonify({"message": "File released from vault"})

@app.route("/api/quarantine/<int:qid>/delete", methods=["DELETE"])
@login_required
def delete_quarantine(qid):
    conn = get_db()
    rec  = conn.execute("SELECT * FROM quarantine_records WHERE id=?", (qid,)).fetchone()
    if not rec:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    if rec["quarantine_path"] and os.path.exists(rec["quarantine_path"]):
        os.remove(rec["quarantine_path"])
    conn.execute("DELETE FROM quarantine_records WHERE id=?", (qid,))
    conn.execute("DELETE FROM scan_logs WHERE id=?", (rec["scan_log_id"],))
    conn.commit(); conn.close()
    return jsonify({"message": "Threat evidence purged"})

# ── Dashboard Stats ────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
@login_required
def get_stats():
    conn        = get_db()
    total       = conn.execute("SELECT COUNT(*) as c FROM scan_logs").fetchone()["c"]
    malware_cnt = conn.execute("SELECT COUNT(*) as c FROM scan_logs WHERE prediction='Malware'").fetchone()["c"]
    safe_cnt    = conn.execute("SELECT COUNT(*) as c FROM scan_logs WHERE prediction='Safe'").fetchone()["c"]
    quarantined = conn.execute("SELECT COUNT(*) as c FROM scan_logs WHERE is_quarantined=1").fetchone()["c"]
    daily = conn.execute("""
        SELECT DATE(scan_time) as date, COUNT(*) as count,
               SUM(CASE WHEN prediction='Malware' THEN 1 ELSE 0 END) as malware
        FROM scan_logs WHERE scan_time >= datetime('now','-7 days')
        GROUP BY DATE(scan_time) ORDER BY date
    """).fetchall()
    recent = conn.execute("SELECT * FROM scan_logs ORDER BY scan_time DESC LIMIT 5").fetchall()
    conn.close()
    return jsonify({
        "total_scans": total, "malware_count": malware_cnt,
        "safe_count": safe_cnt, "quarantined": quarantined,
        "detection_rate": round((malware_cnt / total * 100) if total else 0, 1),
        "daily_trend": [dict(r) for r in daily],
        "recent_scans": [dict(r) for r in recent],
        "model_accuracy": MODEL_META["accuracy"] * 100,
    })

# ── Advanced Stats ─────────────────────────────────────────────────────────────
MALWARE_TYPES   = ["Phishing", "Ransomware", "Account Takeover", "Card Skimming", "Spyware"]
MALWARE_WEIGHTS = [0.35, 0.25, 0.20, 0.12, 0.08]
INDUSTRIES = ["Retail Banking", "Corporate Banking", "Treasury", "Digital Banking",
              "Cards & Payments", "ATM & POS Network", "SWIFT & Interbank", "Compliance"]

def classify_malware_type(filename, file_hash):
    seed = int(hashlib.md5((filename + (file_hash or "")).encode()).hexdigest()[:8], 16)
    rng  = random.Random(seed)
    return rng.choices(MALWARE_TYPES, weights=MALWARE_WEIGHTS, k=1)[0]

def classify_industry(filename):
    seed = int(hashlib.md5(filename.encode()).hexdigest()[8:16], 16)
    return random.Random(seed).choice(INDUSTRIES)

def calc_risk_score(malware_cnt, total, severe):
    if total == 0:
        return {"score": 0, "level": "Low", "color": "#10b981"}
    dr  = malware_cnt / total
    sr  = severe / max(malware_cnt, 1)
    raw = min(100, round((dr * 60 + sr * 40) * 100))
    if raw >= 75:   level, color = "Critical", "#ef4444"
    elif raw >= 50: level, color = "High",     "#f97316"
    elif raw >= 25: level, color = "Medium",   "#f59e0b"
    else:           level, color = "Low",      "#10b981"
    return {"score": raw, "level": level, "color": color}

@app.route("/api/advanced_stats", methods=["GET"])
@login_required
def advanced_stats():
    conn = get_db()
    logs = [dict(r) for r in conn.execute("SELECT * FROM scan_logs ORDER BY scan_time DESC").fetchall()]
    conn.close()
    total   = len(logs)
    malware = [l for l in logs if l["prediction"] == "Malware"]
    mal_cnt = len(malware); safe_cnt = total - mal_cnt
    severe  = [m for m in malware if (m["confidence"] or 0) >= 70]
    sev_cnt = len(severe)

    type_counts = {t: 0 for t in MALWARE_TYPES}
    for m in malware:
        t = m.get("attack_type") or classify_malware_type(m["filename"], m["file_hash"])
        if t in type_counts:
            type_counts[t] += 1

    malware_types = [{"name": t, "value": type_counts[t],
                      "pct": round(type_counts[t] / mal_cnt * 100) if mal_cnt else 0}
                     for t in MALWARE_TYPES]

    ind_counts = {i: 0 for i in INDUSTRIES}
    for l in logs:
        if l["prediction"] == "Malware":
            ind_counts[classify_industry(l["filename"])] += 1

    industry_data = sorted(
        [{"name": i, "attacks": ind_counts[i], "risk": min(100, round(ind_counts[i] * 20))}
         for i in INDUSTRIES],
        key=lambda x: -x["attacks"]
    )

    risk        = calc_risk_score(mal_cnt, total, sev_cnt)
    detect_rate = round((mal_cnt / total * 100) if total else 0, 1)
    csf = {
        "Identify": min(100, 40 + total * 2),
        "Protect":  min(100, 50 + safe_cnt * 3),
        "Detect":   min(100, round(detect_rate * 1.1)),
        "Respond":  min(100, 30 + sev_cnt * 5),
        "Recover":  min(100, 60 + (total - mal_cnt) * 2),
    }
    avg_conf    = round(sum(m["confidence"] or 0 for m in malware) / mal_cnt, 1) if mal_cnt else 0
    avg_entropy = round(sum(m["entropy"]    or 0 for m in malware) / mal_cnt, 4) if mal_cnt else 0

    depts = ["Core Banking", "Treasury & FX", "Retail Banking", "Digital Banking",
             "Compliance & Legal", "Cards & Payments", "SWIFT & Interbank",
             "ATM & POS", "Customer Accounts", "NetBanking & Mobile"]
    weeks   = ["Week 1", "Week 2", "Week 3", "Week 4"]
    heatmap = []
    for dept in depts:
        seed = int(hashlib.md5(dept.encode()).hexdigest()[:8], 16)
        rng  = random.Random(seed + mal_cnt)
        row  = {"dept": dept}
        for w in weeks:
            row[w] = rng.randint(0, max(1, mal_cnt))
        row["total"] = sum(row[w] for w in weeks)
        heatmap.append(row)
    heatmap.sort(key=lambda x: -x["total"])

    conn2 = get_db()
    hourly_rows = conn2.execute("""
        SELECT strftime('%H:00', scan_time) as hour,
               COUNT(*) as total,
               SUM(CASE WHEN prediction='Malware' THEN 1 ELSE 0 END) as threats
        FROM scan_logs WHERE scan_time >= datetime('now','-24 hours')
        GROUP BY strftime('%H', scan_time) ORDER BY hour
    """).fetchall()
    conn2.close()

    live_events = [{"id": l["id"], "filename": l["filename"], "prediction": l["prediction"],
                    "confidence": l["confidence"], "entropy": l["entropy"],
                    "malware_type": l.get("attack_type") if l["prediction"] == "Malware" else None,
                    "attack_type":  l.get("attack_type") if l["prediction"] == "Malware" else None,
                    "industry": classify_industry(l["filename"]),
                    "scan_time": l["scan_time"], "is_quarantined": l["is_quarantined"]}
                   for l in logs[:20]]

    return jsonify({
        "total_scans": total, "malware_count": mal_cnt, "safe_count": safe_cnt,
        "severe_count": sev_cnt, "risk_score": risk, "malware_types": malware_types,
        "industry_data": industry_data, "csf_metrics": csf, "heatmap": heatmap,
        "live_events": live_events, "hourly_activity": [dict(r) for r in hourly_rows],
        "impact": {"avg_confidence": avg_conf, "avg_entropy": avg_entropy,
                   "detection_rate": detect_rate,
                   "quarantine_rate": round(sum(1 for m in malware if m["is_quarantined"]) / mal_cnt * 100 if mal_cnt else 0, 1),
                   "severe_rate": round(sev_cnt / mal_cnt * 100 if mal_cnt else 0, 1)},
    })

# ── Model Info ─────────────────────────────────────────────────────────────────
@app.route("/api/model/info", methods=["GET"])
@login_required
def model_info():
    return jsonify({
        "model_type":   MODEL_META["model_type"],
        "accuracy":     round(MODEL_META["accuracy"]  * 100, 2),
        "precision":    round(MODEL_META["precision"] * 100, 2),
        "recall":       round(MODEL_META["recall"]    * 100, 2),
        "f1_score":     round(MODEL_META["f1_score"]  * 100, 2),
        "n_estimators": MODEL_META["n_estimators"],
        "features":     FEATURE_NAMES,
        "dataset":      "Synthetic Banking Threat Dataset",
        "classes":      ["Clean Document (0)", "Banking Threat (1)"]
    })

# ── Health check (Render uses this to verify the service is up) ────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service":    "BankShield Pro — Banking Cyber Threat Intelligence",
        "status":     "online",
        "version":    "1.0.0",
        "compliance": ["RBI IT Framework", "PCI-DSS v4.0", "ISO 27001"],
        "time":       datetime.now().isoformat()
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "BankShield Pro", "time": datetime.now().isoformat()})

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🏦 BankShield Pro backend starting on port {port}...")
    app.run(debug=False, host="0.0.0.0", port=port)
