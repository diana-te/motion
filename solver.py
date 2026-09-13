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
    mask = crop.max(axis=-1) > 90
    cnt = mask.sum()
    if cnt > 15:
        ys, xs = np.where(mask)
        return xs.mean(), ys.mean(), cnt
    return np.nan, np.nan, 0

def extract_f(crops):
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
    
    return {
        'zeros': zeros,
        'a_std': a_std,
        'span_x': span_x,
        'span_y': span_y
    }

def get_style(f, is_prompt=False):
    if f['zeros'] >= 3 or (f['zeros'] >= 1 and f['a_std'] > 0.55):
        return 'BLINK'
    
    sx, sy = f['span_x'], f['span_y']
    if is_prompt:
        if sx > 3.0 and sx > 2.0 * sy:
            return 'SLIDE_H'
        elif sy > 2.5 and sy > 1.8 * sx:
            return 'SLIDE_V'
    else:
        if sx > 15.0 and sx > 2.0 * sy:
            return 'SLIDE_H'
        elif sy > 8.0 and sy > 2.0 * sx:
            return 'SLIDE_V'
            
    if f['a_std'] >= 0.18:
        return 'PULSE'
        
    return 'CIRCLE_2D'

def solve_motion_captcha(gif_bytes):
    im = Image.open(io.BytesIO(gif_bytes))
    frames = [np.array(im.seek(idx) or im.convert('RGB')) for idx in range(im.n_frames)]
    
    pf = extract_f([fr[PROMPT_BBOX[0]:PROMPT_BBOX[1], PROMPT_BBOX[2]:PROMPT_BBOX[3]] for fr in frames])
    cfs = [extract_f([fr[b[0]:b[1], b[2]:b[3]] for fr in frames]) for b in CELL_BBOXES]
    
    p_style = get_style(pf, is_prompt=True)
    c_styles = [get_style(c, is_prompt=False) for c in cfs]
    
    matches = [c_idx for c_idx, s in enumerate(c_styles) if s == p_style]
    
    # Fallback jika tidak ada yang cocok sempurna
    if not matches:
        dists = []
        for c_idx, c in enumerate(cfs):
            d = abs(c['span_x'] - pf['span_x']) + abs(c['span_y'] - pf['span_y']) + abs(c['a_std'] - pf['a_std']) * 10
            dists.append((c_idx, d))
        dists.sort(key=lambda x: x[1])
        matches = [dists[0][0]]
        
    return p_style, matches
