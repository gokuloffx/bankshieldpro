# 🏦 BankShield Pro — Render.com Deployment Guide

## Folder Structure (Upload this entire folder to GitHub)
```
bankshield_render/
├── app.py                          ← Main Flask app (Render entry point)
├── requirements.txt                ← Python dependencies + gunicorn
├── render.yaml                     ← Render auto-deploy config
├── ml_model/
│   ├── random_forest_model.pkl     ← Trained ML model
│   ├── scaler.pkl                  ← Feature scaler
│   ├── model_metadata.json         ← Model accuracy & feature names
│   └── train_model.py              ← Re-train script (optional)
└── README_RENDER.md                ← This file
```

---

## Step-by-Step Render Deployment

### Step 1 — Push to GitHub
1. Create a new GitHub repo (e.g. `bankshield-pro-backend`)
2. Upload this entire folder to the repo root
3. Make sure `ml_model/*.pkl` files are committed (they are ~2MB)

### Step 2 — Create Render Web Service
1. Go to → https://render.com → Sign up / Login
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in settings:
   - **Name:** `bankshield-pro`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
5. Click **"Create Web Service"**

### Step 3 — Set Environment Variables (in Render Dashboard)
| Key | Value |
|-----|-------|
| `SECRET_KEY` | any random string (e.g. `bankshield_rbi_2024_abc123`) |
| `FRONTEND_URL` | Your React frontend URL (or `*` for open access) |

### Step 4 — Get Your API URL
After deploy, Render gives you a URL like:
```
https://bankshield-pro.onrender.com
```

### Step 5 — Update React Frontend
In your React project, open `frontend/src/api.js` and change:
```js
// Change this:
baseURL: 'http://localhost:5000'
// To this:
baseURL: 'https://bankshield-pro.onrender.com'
```

---

## API Endpoints Available After Deploy

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check — shows service info |
| `/api/health` | GET | Simple health ping |
| `/api/auth/login` | POST | Login |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/me` | GET | Current user |
| `/api/scan` | POST | Upload & scan file |
| `/api/simulate_attack` | POST | Simulate banking attack |
| `/api/simulate_multi` | POST | Simulate multiple attacks |
| `/api/logs` | GET | Audit logs |
| `/api/quarantine` | GET | Threat Vault records |
| `/api/quarantine/:id/restore` | POST | Release from vault |
| `/api/quarantine/:id/delete` | DELETE | Purge threat |
| `/api/stats` | GET | Dashboard stats |
| `/api/advanced_stats` | GET | Banking sector analytics |
| `/api/model/info` | GET | ML model details |

---

## ⚠️ Important Notes for Render Free Tier

1. **Sleep after inactivity:** Free tier sleeps after 15 mins. First request takes ~30s to wake up.
2. **Ephemeral storage:** `/tmp` resets on redeploy — DB and quarantine files are temporary.
3. **For persistent DB:** Upgrade to Render paid tier or use PostgreSQL add-on.
4. **ML model files (.pkl):** These MUST be in GitHub repo — do not gitignore them.

---

## Default Login Credentials
- **Username:** `admin`
- **Password:** `admin123`

*BankShield Pro — Protecting Indian Banking Infrastructure* 🏦
