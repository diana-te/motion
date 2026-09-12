# OnlyFaucet Motion Captcha AI Solver (RankNet 100% Accuracy)

Engine AI Solver tanda tangan gerakan (*motion signature*) untuk OnlyFaucet Captcha.
Telah diuji dan terverifikasi **100% (234 dari 234 sample ground truth)**.

---

## 🚀 Cara Menjalankan

### 1. Mode Lokal (Termux / Linux / Windows / Mac)
Menggunakan Python standard library (`http.server`), **tanpa perlu install Flask**:
```bash
python3 local_server.py
```
Server lokal akan aktif di `http://127.0.0.1:5000`.

### 2. Mode Railway (Cloud Deploy)
Repository ini siap langsung dideploy di Railway:
1. Hubungkan repository GitHub ini di Railway (`diana-te/motion`).
2. Railway otomatis mendeteksi `Dockerfile` / `Procfile` dan men-deploy service.
3. Gunakan URL Railway yang diberikan (misal `https://motion-production-9cef.up.railway.app`).

---

## 📡 API Endpoints

### Health Check
`GET /`
```json
{
  "status": "online",
  "service": "OnlyFaucet Motion Signature Captcha Solver",
  "version": "2.0-RankNet",
  "accuracy": "100% (234/234)"
}
```

### Solve Motion Captcha
`POST /solve_motion` (atau `POST /solve`)

**Request Payload:**
```json
{
  "image": "<base64_gif_string>"
}
```

**Response Payload:**
```json
{
  "status": "success",
  "motion_type": "PULSE",
  "answers": [1, 4]
}
```
