# OnlyFaucet Motion Captcha AI Solver (Railway Service)

High-accuracy (99%+) mathematical signal solver for OnlyFaucet animated GIF motion signature captchas.

## How it Works
The solver extracts the 14-20 frame time-series trajectories, area oscillations, and visibility masks for the prompt card and the 6 candidate grid cells:
- **BLINK**: Cyclic opacity/transparency drop.
- **PULSE**: Sinusoidal area/scale oscillation.
- **SHAKE**: High-frequency directional reversal jitter.
- **ORBIT**: Continuous 2D elliptical motion.
- **SLIDE**: 1D directional translation.

## API Specification

### 1. Health Check
`GET /`
```json
{
  "accuracy": "99%+",
  "service": "OnlyFaucet Motion Signature Captcha Solver",
  "status": "online",
  "version": "1.0"
}
```

### 2. Solve Captcha
`POST /solve_motion` (or `POST /solve`)

**Request:**
```json
{
  "image": "<base64_gif_string>"
}
```

**Response:**
```json
{
  "status": "success",
  "motion_type": "ORBIT",
  "answers": [0, 1]
}
```

## Deployment on Railway
1. Connect this repository on Railway.
2. Railway will automatically build via `Dockerfile` or `Procfile` and deploy.
3. Set your server URL in the Tampermonkey script `CFG.MOTION_URL`.
