import io
import numpy as np
from PIL import Image

CELL_BBOXES = [
    (104, 248, 8, 196),    # 0 (Row 0, Col 0)
    (104, 248, 206, 394),  # 1 (Row 0, Col 1)
    (104, 248, 404, 592),  # 2 (Row 0, Col 2)
    (258, 402, 8, 196),    # 3 (Row 1, Col 0)
    (258, 402, 206, 394),  # 4 (Row 1, Col 1)
    (258, 402, 404, 592),  # 5 (Row 1, Col 2)
]
PROMPT_BBOX = (13, 83, 481, 579)

def get_char_metrics(crop):
    mask = crop.max(axis=-1) > 120
    cnt = mask.sum()
    if cnt > 15:
        ys, xs = np.where(mask)
        return xs.mean(), ys.mean(), cnt
    return np.nan, np.nan, 0

def get_feat(crops):
    xs, ys, areas = [], [], []
    for c in crops:
        x, y, a = get_char_metrics(c)
        xs.append(x)
        ys.append(y)
        areas.append(a)
    xs = np.array(xs)
    ys = np.array(ys)
    areas = np.array(areas, dtype=float)
    
    zeros = np.sum(areas < 50)
    a_mean = np.mean(areas) + 1e-5
    a_std = np.std(areas) / a_mean
    
    med_x = np.nanmedian(xs) if np.sum(~np.isnan(xs)) > 0 else 0
    med_y = np.nanmedian(ys) if np.sum(~np.isnan(ys)) > 0 else 0
    cx = np.where(np.isnan(xs), med_x, xs)
    cy = np.where(np.isnan(ys), med_y, ys)
    
    span_x = np.max(cx) - np.min(cx)
    span_y = np.max(cy) - np.min(cy)
    
    acc_x = np.diff(cx, 2)
    acc_y = np.diff(cy, 2)
    jitter = np.mean(acc_x**2 + acc_y**2)
    
    flips_x = np.sum(np.diff(np.sign(np.diff(cx)[np.abs(np.diff(cx)) > 0.3])) != 0)
    flips_y = np.sum(np.diff(np.sign(np.diff(cy)[np.abs(np.diff(cy)) > 0.3])) != 0)
    flips = flips_x + flips_y
    
    return {
        'zeros': zeros,
        'a_std': a_std,
        'span_x': span_x,
        'span_y': span_y,
        'jitter': jitter,
        'flips': flips,
        'areas': areas,
        'cx': cx,
        'cy': cy
    }

def solve_motion_captcha(gif_bytes):
    im = Image.open(io.BytesIO(gif_bytes))
    frames = [np.array(im.seek(i) or im.convert('RGB')) for i in range(im.n_frames)]
    
    pf = get_feat([f[PROMPT_BBOX[0]:PROMPT_BBOX[1], PROMPT_BBOX[2]:PROMPT_BBOX[3]] for f in frames])
    cfs = [get_feat([f[b[0]:b[1], b[2]:b[3]] for f in frames]) for b in CELL_BBOXES]
    
    is_blink = pf['zeros'] >= 3 or (pf['zeros'] >= 1 and pf['a_std'] > 0.5)
    is_pulse = pf['a_std'] > 0.18 and not is_blink
    
    if is_blink:
        mtype = 'BLINK'
    elif is_pulse:
        mtype = 'PULSE'
    elif (pf['span_x'] >= 6 and pf['span_y'] >= 6) and (pf['flips'] >= 8 or pf['jitter'] > 0.5):
        mtype = 'SHAKE'
    else:
        mtype = 'ORBIT'
    
    scores = []
    for i in range(6):
        for j in range(i+1, 6):
            ci, cj = cfs[i], cfs[j]
            score = 0.0
            
            if is_blink:
                score += (ci['zeros'] + cj['zeros']) * 15 - abs(ci['zeros'] - cj['zeros']) * 10
                if ci['zeros'] < 2 or cj['zeros'] < 2:
                    score -= 500
            elif is_pulse:
                if ci['a_std'] < 0.15 or cj['a_std'] < 0.15:
                    score -= 1000
                score += 100 - abs(ci['a_std'] - cj['a_std']) * 80
                score -= (abs(ci['a_std'] - pf['a_std']) + abs(cj['a_std'] - pf['a_std'])) * 20
                ci_corr = np.corrcoef(pf['areas'], ci['areas'])[0, 1] if np.std(ci['areas']) > 0 else 0
                cj_corr = np.corrcoef(pf['areas'], cj['areas'])[0, 1] if np.std(cj['areas']) > 0 else 0
                score += (abs(ci_corr) + abs(cj_corr)) * 40
                if ci['zeros'] >= 5 or cj['zeros'] >= 5:
                    score -= 500
            else:
                if ci['a_std'] > 0.15 or cj['a_std'] > 0.15 or ci['zeros'] > 0 or cj['zeros'] > 0:
                    score -= 1000
                
                is_shake = (pf['span_x'] >= 6 and pf['span_y'] >= 6) and (pf['flips'] >= 8 or pf['jitter'] > 0.5)
                if is_shake:
                    score += (ci['flips'] + cj['flips']) * 5 + (ci['jitter'] + cj['jitter']) * 5
                    score -= abs(ci['span_x'] - pf['span_x']) * 3 + abs(cj['span_x'] - pf['span_x']) * 3
                else:
                    score += 100 - (ci['flips'] + cj['flips']) * 10 - (ci['jitter'] + cj['jitter']) * 2
                    score -= abs(ci['flips'] - cj['flips']) * 15
                    score -= abs(ci['span_x'] - cj['span_x']) * 2 + abs(ci['span_y'] - cj['span_y']) * 2
                    
            scores.append(((i, j), score))
            
    scores.sort(key=lambda x: -x[1])
    best_pair = list(scores[0][0])
    return mtype, best_pair
