# NSE Scanner – Deployment Guide

## Files
```
nse-scanner/
├── app.py                          ← Flask backend (all logic)
├── templates/index.html            ← Mobile-responsive UI
├── requirements.txt                ← Python dependencies
├── Procfile                        ← For Render/Heroku
├── render.yaml                     ← Render auto-config
└── nse_derivative_stocks_FullList.csv  ← Your CSV (add this!)
```

---

## Step 1 – Push to GitHub

1. Create a free account at https://github.com
2. Create a new **private** repository named `nse-scanner`
3. Upload all files in this folder to the repository
4. **Also upload your** `nse_derivative_stocks_FullList.csv`

```bash
# Or use git from terminal:
git init
git add .
git commit -m "Initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/nse-scanner.git
git push -u origin main
```

---

## Step 2 – Deploy on Render (Free)

1. Go to https://render.com and sign up (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account and select `nse-scanner` repo
4. Render auto-detects settings from `render.yaml`
5. Click **Deploy**
6. Your app will be live at: `https://nse-scanner.onrender.com`

---

## Step 3 – Update Access Token Daily

Since Kite access tokens expire daily, update it each morning:

### Option A: Render Dashboard (Easiest)
1. Go to https://dashboard.render.com
2. Open your `nse-scanner` service
3. Click **Environment** tab
4. Update `KITE_ACCESS_TOKEN` value
5. Click **Save** — app restarts automatically in ~30 seconds

### Option B: Update in app.py before push
Edit line in `app.py`:
```python
ACCESS_TOKEN = os.environ.get("KITE_ACCESS_TOKEN", "YOUR_NEW_TOKEN_HERE")
```

---

## Free Tier Limitations (Render)

| Feature | Free Tier |
|---------|-----------|
| Sleep after inactivity | Yes (15 min) |
| Wake-up time | ~30 seconds |
| Monthly hours | 750 hrs |
| Custom domain | Yes (free) |

**Tip:** Since you use this during market hours (9:15–3:30), just open the URL
before trading starts. The first load wakes the server (30 sec), then it stays
awake while you're using it.

---

## Mobile Usage

- Open the URL in Chrome/Safari on your phone
- Add to home screen: Share → "Add to Home Screen" (works like an app)
- Cards view loads automatically on mobile
- Tap any stock symbol to open TradingView chart

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `Invalid access token` | Update `KITE_ACCESS_TOKEN` in Render env vars |
| `Module not found` | Check `requirements.txt` has all packages |
| App sleeping | Just reload — it wakes in 30 sec |
| CSV not found | Upload `nse_derivative_stocks_FullList.csv` to repo root |
