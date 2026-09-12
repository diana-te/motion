import io
import numpy as np
from PIL import Image

MEAN = np.array([0.9712279049222613, 0.5722181628193076, 0.8332015019570103, 0.3345763518319441, 0.9091396356137667, 0.3473415536864585, 2.0686274509803924, 1.6294117647058823, 0.40556056228874077, 0.2765270809562815, 2.991902465807682, 2.3091685643313355, 8.084269489990616, 6.23776000910627, 6.765468421581352, 4.799916814795093, 18.04901960784314, 6.182352941176471, 105.58563568159735, 81.58513672636064])
STD = np.array([0.4282453363173764, 0.2841610498828866, 0.34425028781913386, 0.20047995395072024, 0.3213523396982652, 0.19564811286027778, 2.5090427041117422, 2.1305255711084623, 0.28897418411891096, 0.23601838498604294, 2.4260869978383877, 2.0600339698181105, 10.115127787160901, 10.083456460351842, 7.673992209589366, 7.208341713958787, 8.457249255693993, 4.576591300364816, 206.56514336933586, 191.45160506340824])
WEIGHTS = np.array([1.6861404090650738, 1.719301421575001, 0.7218431367657517, 0.9429251404233943, 1.054459084554888, 2.031841245861182, 0.6584817386445249, 0.10065777921235397, 1.0711266042741099, 1.206949249281629, 1.2029200028270632, 1.946402109398635, -0.6339909534978861, -0.7665255499413144, 0.6028873828555036, -1.0288035001497546, 0.9685271415963563, 2.2469680849517477, 0.01717774425331129, -0.04863268881143459])

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

def get_fft(sig):
    s = sig - np.nanmean(sig)
    if np.nanstd(s) < 1e-4:
        return np.zeros(len(sig)//2 + 1)
    s = np.nan_to_num(s, nan=0.0)
    mag = np.abs(np.fft.rfft(s))
    return mag / (np.sum(mag) + 1e-5)

def extract_single(crops):
    xs, ys, areas = [], [], []
    for c in crops:
        x, y, a = get_char_metrics(c)
        xs.append(x)
        ys.append(y)
        areas.append(a)
    xs = np.array(xs)
    ys = np.array(ys)
    areas = np.array(areas, dtype=float)
    
    med_x = np.nanmedian(xs) if np.sum(~np.isnan(xs)) > 0 else 0
    med_y = np.nanmedian(ys) if np.sum(~np.isnan(ys)) > 0 else 0
    cx = np.where(np.isnan(xs), med_x, xs)
    cy = np.where(np.isnan(ys), med_y, ys)
    
    span_x = np.max(cx) - np.min(cx)
    span_y = np.max(cy) - np.min(cy)
    
    zeros = np.sum(areas < 50)
    a_mean = np.mean(areas) + 1e-5
    a_std = np.std(areas) / a_mean
    a_amp = np.log((np.max(areas) + 1.0) / (np.min(areas) + 1.0))
    
    acc_x = np.diff(cx, 2)
    acc_y = np.diff(cy, 2)
    jitter = np.mean(acc_x**2 + acc_y**2)
    
    flips_x = np.sum(np.diff(np.sign(np.diff(cx)[np.abs(np.diff(cx))>0.2]))!=0)
    flips_y = np.sum(np.diff(np.sign(np.diff(cy)[np.abs(np.diff(cy))>0.2]))!=0)
    
    a_spec = get_fft(areas)
    x_spec = get_fft(cx)
    y_spec = get_fft(cy)
    
    return {
        'zeros': zeros,
        'a_std': a_std,
        'a_amp': a_amp,
        'span_x': span_x,
        'span_y': span_y,
        'jitter': jitter,
        'flips': flips_x + flips_y,
        'a_spec': a_spec,
        'x_spec': x_spec,
        'y_spec': y_spec
    }

def make_pair_features(pf, ci, cj):
    def d(f_name):
        return [
            abs(ci[f_name] - pf[f_name]) + abs(cj[f_name] - pf[f_name]),
            abs(ci[f_name] - cj[f_name])
        ]
        
    def d_vec(f_name):
        return [
            np.linalg.norm(ci[f_name] - pf[f_name]) + np.linalg.norm(cj[f_name] - pf[f_name]),
            np.linalg.norm(ci[f_name] - cj[f_name])
        ]
        
    feats = []
    feats.extend(d_vec('a_spec'))
    feats.extend(d_vec('x_spec'))
    feats.extend(d_vec('y_spec'))
    feats.extend(d('zeros'))
    feats.extend(d('a_std'))
    feats.extend(d('a_amp'))
    feats.extend(d('span_x'))
    feats.extend(d('span_y'))
    feats.extend(d('flips'))
    feats.extend(d('jitter'))
    
    return np.array(feats, dtype=float)

def solve_motion_captcha(gif_bytes):
    im = Image.open(io.BytesIO(gif_bytes))
    frames = [np.array(im.seek(i) or im.convert('RGB')) for i in range(im.n_frames)]
    
    pf = extract_single([f[PROMPT_BBOX[0]:PROMPT_BBOX[1], PROMPT_BBOX[2]:PROMPT_BBOX[3]] for f in frames])
    cfs = [extract_single([f[b[0]:b[1], b[2]:b[3]] for f in frames]) for b in CELL_BBOXES]
    
    if pf['zeros'] >= 3 or (pf['zeros'] >= 1 and pf['a_std'] > 0.55):
        mtype = 'BLINK'
    elif pf['a_std'] >= 0.18:
        mtype = 'PULSE'
    elif (pf['span_x'] >= 5 and pf['span_y'] >= 5) and (pf['flips'] >= 8 or pf['jitter'] > 0.5):
        mtype = 'SHAKE'
    elif (pf['span_x'] >= 3.0 or pf['span_y'] >= 3.0):
        mtype = 'ORBIT'
    else:
        mtype = 'STATIC'
        
    pair_scores = []
    for i in range(6):
        for j in range(i+1, 6):
            f_vec = make_pair_features(pf, cfs[i], cfs[j])
            f_norm = (f_vec - MEAN) / STD
            score = - np.dot(f_norm, WEIGHTS)
            pair_scores.append(((i, j), score))
            
    pair_scores.sort(key=lambda x: -x[1])
    best_pair = list(pair_scores[0][0])
    return mtype, best_pair
