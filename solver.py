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
        return xs.mean(), ys.mean(), cnt, 1.0
    return np.nan, np.nan, 0, 0.0

def extract_features(m_list):
    xs = np.array([m[0] for m in m_list])
    ys = np.array([m[1] for m in m_list])
    areas = np.array([m[2] for m in m_list])
    vis = np.array([m[3] for m in m_list])
    
    med_area = np.median(areas[areas > 0]) if np.sum(areas > 0) > 0 else 1
    low_frames = np.sum(areas < 0.25 * med_area)
    is_blink = low_frames >= 3
    
    norm_areas = areas[areas >= 0.25 * med_area]
    pulse_ratio = (np.std(norm_areas) / (np.mean(norm_areas) + 1e-5)) if len(norm_areas) > 2 else 0.0
    is_pulse = (pulse_ratio > 0.30) and not is_blink
    
    med_x = np.nanmedian(xs) if np.sum(vis) > 0 else 0
    med_y = np.nanmedian(ys) if np.sum(vis) > 0 else 0
    cx = np.where(np.isnan(xs), med_x, xs)
    cy = np.where(np.isnan(ys), med_y, ys)
    
    span_x = np.max(cx) - np.min(cx)
    span_y = np.max(cy) - np.min(cy)
    
    dx = np.diff(cx)
    dy = np.diff(cy)
    flips_x = np.sum(np.diff(np.sign(dx[np.abs(dx) > 0.4])) != 0)
    flips_y = np.sum(np.diff(np.sign(dy[np.abs(dy) > 0.4])) != 0)
    total_flips = flips_x + flips_y
    
    is_shake = (span_x >= 6 and span_y >= 6) and (total_flips >= 10)
    is_slide_h = (span_x > 20 and span_y < 8)
    is_slide_v = (span_y > 20 and span_x < 8)
    is_orbit = not is_blink and not is_pulse and not is_shake and not is_slide_h and not is_slide_v
    
    return {
        'is_blink': is_blink,
        'is_pulse': is_pulse,
        'pulse_ratio': pulse_ratio,
        'is_shake': is_shake,
        'is_slide_h': is_slide_h,
        'is_slide_v': is_slide_v,
        'is_orbit': is_orbit,
        'span_x': span_x,
        'span_y': span_y,
        'flips': total_flips
    }

def solve_motion_captcha(gif_bytes):
    im = Image.open(io.BytesIO(gif_bytes))
    frames = [np.array(im.seek(i) or im.convert('RGB')) for i in range(im.n_frames)]
    
    p_m = [get_char_metrics(f[PROMPT_BBOX[0]:PROMPT_BBOX[1], PROMPT_BBOX[2]:PROMPT_BBOX[3]]) for f in frames]
    pf = extract_features(p_m)
    cfs = [extract_features([get_char_metrics(f[b[0]:b[1], b[2]:b[3]]) for f in frames]) for b in CELL_BBOXES]
    
    if pf['is_blink']:
        p_type = 'BLINK'
    elif pf['is_pulse']:
        p_type = 'PULSE'
    elif pf['is_shake']:
        p_type = 'SHAKE'
    elif pf['is_slide_h']:
        p_type = 'SLIDE_H'
    elif pf['is_slide_v']:
        p_type = 'SLIDE_V'
    else:
        p_type = 'ORBIT'
        
    scores = []
    for i in range(6):
        for j in range(i+1, 6):
            fi, fj = cfs[i], cfs[j]
            score = 0.0
            if p_type == 'BLINK':
                score = (10.0 if fi['is_blink'] else -10.0) + (10.0 if fj['is_blink'] else -10.0)
            elif p_type == 'PULSE':
                score = (10.0 if fi['is_pulse'] else -10.0) + (10.0 if fj['is_pulse'] else -10.0)
                score += (fi['pulse_ratio'] + fj['pulse_ratio']) * 5 - abs(fi['pulse_ratio'] - fj['pulse_ratio']) * 5
            elif p_type == 'SHAKE':
                score = (10.0 if fi['is_shake'] else -10.0) + (10.0 if fj['is_shake'] else -10.0)
                score += (fi['flips'] + fj['flips']) - abs(fi['flips'] - fj['flips'])
            elif p_type == 'ORBIT':
                score = (10.0 if fi['is_orbit'] else -10.0) + (10.0 if fj['is_orbit'] else -10.0)
                if fi['is_blink'] or fj['is_blink']: score -= 20
                if fi['is_pulse'] or fj['is_pulse']: score -= 20
                if fi['is_shake'] or fj['is_shake']: score -= 20
            elif p_type in ('SLIDE_H', 'SLIDE_V'):
                key = 'is_slide_h' if p_type == 'SLIDE_H' else 'is_slide_v'
                score = (10.0 if fi[key] else -10.0) + (10.0 if fj[key] else -10.0)
                
            scores.append(((i, j), score))
            
    scores.sort(key=lambda x: -x[1])
    best_pair = list(scores[0][0])
    return p_type, best_pair
