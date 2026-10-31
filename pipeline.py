#!/usr/bin/env python3
"""
Multi-emotion recognition in song lyrics with symbolic melody features and in-domain transfer learning.
One file, one command per stage.  See README.md.

    python pipeline.py all --data-dir data --out-dir results           # everything, in order
    python pipeline.py prepare|finetune|embed|song_nn|song_linear|verse_level|text_ablation|summary ...

Stages
  prepare        XML + CSVs -> songs.pkl (100 songs: text, labels, semitone intervals, melody features), verses.pkl (4,9xx lines)
  finetune       BERT-base fine-tuned on EmotionWheel (8 labels)                         [GPU recommended]
  embed          [CLS] and mean-pooled vectors: fine-tuned vs plain bert-base-uncased    [GPU recommended]
  song_nn        Text-only / Music-only (BiLSTM) / Attention Fusion, 5-fold CV x seeds   [GPU recommended]
  song_linear    all-positive baseline, linear models on text + melody features, univariate tests, power analysis
  verse_level    line-level regression + classification, GroupKFold (by song) vs plain KFold (leakage check)
  text_ablation  fine-tuned vs plain BERT (CLS / mean-pool), logistic regression and MLP
  summary        collects every result table into SUMMARY.md
All analysis settings below were fixed before the corresponding experiments were run; nothing is tuned on test folds.
"""
import argparse, glob, json, os, re, sys, time, warnings
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
EMO = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
EW_COLS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']
BASE_PC = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
DEG = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11}
PAD = 99.0          # sentinel padding value for interval sequences (true values lie in [-12, 12])
THR_NN = 0.3        # decision threshold for the neural models (as in the thesis)


# ----------------------------------------------------------------------------- paths / io
class Paths:
    def __init__(self, a):
        self.data, self.out = a.data_dir, a.out_dir
        os.makedirs(self.out, exist_ok=True)
    def d(self, *p): return os.path.join(self.data, *p)
    def o(self, *p):
        f = os.path.join(self.out, *p); os.makedirs(os.path.dirname(f), exist_ok=True); return f

def save_table(P, df, name):
    df.to_csv(P.o(name), index=False); print(f'  -> {P.o(name)}'); return df

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ----------------------------------------------------------------------------- stage: prepare
def pitch(note):
    """orig-note such as A, E', Bb, G--, C#'' -> absolute semitone number ( ' = +1 octave, - = -1 octave)."""
    L, acc, o = re.match(r"([A-G])([#b]*)(['\-]*)$", note).groups()
    return BASE_PC[L] + acc.count('#') - acc.count('b') + 12 * o.count("'") - 12 * o.count('-')

def parse_xml(path):
    """Returns {xml_id: {'keys': [...], 'tokens': DataFrame(time, dur, note, pitch, tone), 'verses': [...]}}."""
    x = open(path, encoding='utf-8', errors='ignore').read(); out = {}
    for b in re.split(r'(?=<song )', x)[1:]:
        sid = re.search(r'filename=(\S+?)>', b[:400]).group(1).split('/')[-1].split('.')[0]
        keys = re.findall(r'<key time=\d+>([^<]*)</key>', b)
        rows, verses, last_tone = [], [], 0
        # split on <verse ...> openings (one verse in LETITSNO has no closing tag, so a <verse>...</verse> regex would drop its notes)
        for chunk in re.split(r'(?=<verse )', b)[1:]:
            attrs = dict(re.findall(r'(\w+)=(\S+)', re.match(r'<verse ([^>]*)>', chunk).group(1))); vt = []
            for attr, txt in re.findall(r'<token ([^>]*)>(.*?)</token>', chunk, flags=re.S):
                note = re.search(r'orig-note=(\S+)', attr).group(1); tone = re.search(r'tone=(\S+)', attr).group(1)
                if tone != 'NIL':   # major-scale degree 1-7 -> semitones above the tonic; '+'/'-' = raised/lowered. NIL: carry forward.
                    last_tone = DEG[int(tone[0])] + (1 if tone.endswith('+') else -1 if tone.endswith('-') else 0)
                rows.append(dict(time=int(re.search(r'time=(\d+)', attr).group(1)), dur=int(re.search(r'interval=(\d+)', attr).group(1)),
                                 note=note, pitch=pitch(note), tone=last_tone)); vt.append((len(rows) - 1, txt))
            if vt: verses.append(dict(attrs=attrs, idx=[i for i, _ in vt], text=''.join(t for _, t in vt).strip()))
        assert len(rows) == len(re.findall(r'<token ', b)), f'{sid}: tokens outside <verse> blocks'
        out[sid] = dict(keys=keys, tokens=pd.DataFrame(rows), verses=verses)
    return out

def melody_features(keys, tk):
    """Pre-declared melody descriptors (see README): interval statistics, rhythm/tempo, mode, pitch height."""
    p, t, d = tk.pitch.values, tk.time.values, tk.dur.values
    iv = np.clip(np.diff(p), -12, 12); a = np.abs(iv); tv = np.clip(np.diff(tk.tone.values), -12, 12)
    h = np.bincount(iv + 12, minlength=25) / len(iv); ent = -(h[h > 0] * np.log2(h[h > 0])).sum()
    ioi = np.diff(t); ok = ioi > 0; ioi_c = np.clip(ioi[ok], 1, 1920); dd, ii = d[:-1][ok], ioi[ok]; sel = (ii >= 120) & (ii <= 960)
    minor = np.array(['minor' in k.lower() for k in keys]); span = max(t.max() - t.min(), 1)
    return dict(
        iv_mean=iv.mean(), iv_std=iv.std(), iv_absmean=a.mean(), frac_zero=(iv == 0).mean(), frac_leap=(a >= 5).mean(),
        frac_up=(iv > 0).mean(), frac_down=(iv < 0).mean(), iv_entropy=ent, log_len=np.log(len(iv) + 1),
        tone_std=tv.std(), tone_absmean=np.abs(tv).mean(),
        ioi_mean=ioi_c.mean(), ioi_cv=ioi_c.std() / ioi_c.mean(), frac_short=(ioi_c <= 240).mean(),
        ms_per_tick=np.median(dd[sel] / ii[sel]) if sel.any() else np.nan, dur_mean=d.mean(), notes_per_beat=len(t) / (span / 480),
        is_minor=float(minor[0]) if len(minor) else np.nan, frac_minor_keys=minor.mean() if len(minor) else np.nan,
        n_key_changes=max(len(keys) - 1, 0),
        pitch_mean=p.mean(), pitch_std=p.std(), pitch_range=np.percentile(p, 95) - np.percentile(p, 5), pitch_p95=np.percentile(p, 95))

GROUPS = {'interval stats': ['iv_mean', 'iv_std', 'iv_absmean', 'frac_zero', 'frac_leap', 'frac_up', 'frac_down', 'iv_entropy', 'log_len', 'tone_std', 'tone_absmean'],
          'rhythm/tempo': ['ioi_mean', 'ioi_cv', 'frac_short', 'ms_per_tick', 'dur_mean', 'notes_per_beat'],
          'mode': ['is_minor', 'frac_minor_keys', 'n_key_changes'],
          'pitch': ['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_p95']}

def stage_prepare(a, P):
    xml = (glob.glob(P.d('Songs*100songs.xml')) + glob.glob(P.d('*.xml')))[0]
    log(f'parsing {xml}')
    S = parse_xml(xml)
    mp = pd.read_csv(P.d('id_mapping.csv')); an = pd.read_csv(P.d('MihalceaNewAnnotation.csv'))
    # IMPORTANT: join by POSITION (checked against titles). Two different songs are both called "Just The Way You Are",
    # so merging on title silently gives one of them the other's lyrics/labels.
    assert len(mp) == len(an) == len(S) == 100, 'expected 100 songs'
    assert (mp.song_name_csv.values == an.Song.values).all(), 'id_mapping.csv and MihalceaNewAnnotation.csv are not in the same order'
    assert set(mp.xml_id) == set(S), 'xml ids in id_mapping.csv do not match the XML'
    rows = []
    for i, r in an.iterrows():
        sid = mp.xml_id[i]; tk = S[sid]['tokens']; iv = np.clip(np.diff(tk.pitch.values), -12, 12).astype(int)
        rows.append(dict(xml_id=sid, song=r.Song, artist=r.Artists, text=r.text, **{e: int(r[e]) for e in EMO},
                         intervals=iv.tolist(), tone_intervals=np.clip(np.diff(tk.tone.values), -12, 12).astype(int).tolist(), seq_len=len(tk),
                         **{'f_' + k: v for k, v in melody_features(S[sid]['keys'], tk).items()}))
    songs = pd.DataFrame(rows); fc = [c for c in songs if c.startswith('f_')]
    songs[fc] = songs[fc].fillna(songs[fc].median())
    songs.to_pickle(P.o('songs.pkl'))
    L = songs.intervals.apply(len)
    print(f'  songs: {len(songs)} | interval length min/mean/max = {L.min()}/{L.mean():.1f}/{L.max()} | labels: {dict(songs[EMO].sum())}')
    # line-level data
    vr = []
    for i, r in an.iterrows():
        sid = mp.xml_id[i]; tk = S[sid]['tokens']; minor = float('minor' in S[sid]['keys'][0].lower()) if S[sid]['keys'] else np.nan
        for k, v in enumerate(S[sid]['verses']):
            sub = tk.iloc[v['idx']]
            vr.append(dict(song=sid, line=k, text=v['text'], n_notes=len(sub), pitch=sub.pitch.values, time=sub.time.values, dur=sub.dur.values,
                           is_minor=minor, **{e: float(v['attrs'].get(e, 0)) for e in EMO}))
    verses = pd.DataFrame(vr); verses = verses[verses.text.str.len() >= 3].reset_index(drop=True)
    verses.to_pickle(P.o('verses.pkl')); print(f'  lines: {len(verses)} in {verses.song.nunique()} songs')
    return songs


# ----------------------------------------------------------------------------- shared helpers
def load_songs(P):
    f = P.o('songs.pkl')
    if not os.path.exists(f): raise SystemExit('songs.pkl missing: run the "prepare" stage first')
    return pd.read_pickle(f)

def load_emb(P):
    f = P.o('embeddings.npz')
    if not os.path.exists(f): raise SystemExit('embeddings.npz missing: run the "embed" stage first')
    return dict(np.load(f))

def macro_micro_auc(Y, prob, thr, idx=None):
    from sklearn.metrics import f1_score, roc_auc_score
    idx = np.arange(len(Y)) if idx is None else idx; y, p = Y[idx], prob[idx]
    per_f1 = np.array([f1_score(y[:, j], p[:, j] > thr, zero_division=0) for j in range(6)])
    auc = [roc_auc_score(y[:, j], p[:, j]) if len(np.unique(y[:, j])) == 2 else np.nan for j in range(6)]   # AUC undefined if a label is constant
    return dict(macro_f1=per_f1.mean(), micro_f1=f1_score(y, p > thr, average='micro', zero_division=0), macro_auc=np.nanmean(auc), per_f1=per_f1)

def fold_ids(n, seed, n_splits=5):
    """Fold index of every song for KFold(shuffle, random_state=seed): the split used by every song-level experiment of that seed."""
    from sklearn.model_selection import KFold
    f = np.zeros(n, int)
    for k, (_, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(np.arange(n))): f[te] = k
    return f

def foldwise(Y, prob, thr, fid, idx=None):
    """Metrics computed inside each test fold and then averaged over folds (AUC of pooled out-of-fold scores is biased for near-constant predictors).
    idx: optional (bootstrap) resample of songs, possibly with repeats; each resampled song is evaluated in the fold it belongs to."""
    idx = np.arange(len(Y)) if idx is None else np.asarray(idx); out = []
    for k in np.unique(fid):
        sub = idx[fid[idx] == k]
        if len(sub): out.append(macro_micro_auc(Y, prob, thr, sub))
    return {m: np.nanmean([o[m] for o in out]) for m in ('macro_f1', 'micro_f1', 'macro_auc')} | {'per_f1': np.mean([o['per_f1'] for o in out], 0)}

def boot_ci(fn, n, B=1000, seed=0):
    """Cluster bootstrap over songs: fn(idx) -> float. Returns mean, 2.5%, 97.5% (resamples where fn fails are skipped)."""
    rng = np.random.default_rng(seed); v = []
    for _ in range(B):
        try: v.append(fn(rng.integers(0, n, n)))
        except ValueError: pass
    v = np.array(v); return v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5), v.std()

def boot_pair(Y, A, B, thr, fid, B_=1000, seed=0):
    """Cluster bootstrap over songs for (A - B) in macro_auc and macro_f1 (both computed fold-wise, see foldwise), one pass over the resamples."""
    rng = np.random.default_rng(seed); n = len(Y); d = {'macro_auc': [], 'macro_f1': []}
    for _ in range(B_):
        i = rng.integers(0, n, n); a, b = foldwise(Y, A, thr, fid, i), foldwise(Y, B, thr, fid, i)
        for k in d: d[k].append(a[k] - b[k])
    return {k: (np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5), np.std(v)) for k, v in d.items()}

def bh(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); q = np.empty(m)
    q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]; return np.clip(q, 0, 1)

def lr_model(C=0.1):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=2000))

def oof_lr(X, Y, seed, n_splits=5):
    """Out-of-fold probabilities of one logistic regression per emotion."""
    from sklearn.model_selection import KFold
    P = np.zeros(Y.shape)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        for j in range(Y.shape[1]):
            if Y[tr, j].min() == Y[tr, j].max(): P[te, j] = float(Y[tr, j][0])
            else: P[te, j] = lr_model().fit(X[tr], Y[tr, j]).predict_proba(X[te])[:, 1]
    return P

def need_torch():
    try:
        import torch; return torch
    except ImportError: raise SystemExit('this stage needs PyTorch (pip install torch transformers)')


# ----------------------------------------------------------------------------- stage: finetune / embed
def stage_finetune(a, P):
    torch = need_torch(); import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import AdamW
    from transformers import BertTokenizer, BertForSequenceClassification
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(42); np.random.seed(42)
    ew = pd.read_csv(P.d('EmotionWheelFinal.csv')); log(f'fine-tuning BERT on {len(ew)} EmotionWheel songs ({EW_COLS})')
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8, problem_type='multi_label_classification').to(dev)
    class DS(Dataset):
        def __init__(s, t, y): s.t, s.y = t, y
        def __len__(s): return len(s.t)
        def __getitem__(s, i):
            e = tok(str(s.t[i]), add_special_tokens=True, max_length=256, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
            return {'input_ids': e['input_ids'].flatten(), 'attention_mask': e['attention_mask'].flatten(), 'labels': torch.FloatTensor(s.y[i])}
    loader = DataLoader(DS(ew['Lyrics'].values, ew[EW_COLS].values), batch_size=16, shuffle=True)
    opt = AdamW(model.parameters(), lr=2e-5); hist = []
    for ep in range(5):
        model.train(); tot = 0
        for b in loader:
            opt.zero_grad(); out = model(b['input_ids'].to(dev), attention_mask=b['attention_mask'].to(dev), labels=b['labels'].to(dev))
            out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += out.loss.item()
        hist.append(dict(epoch=ep + 1, loss=tot / len(loader))); log(f'epoch {ep + 1} loss {hist[-1]["loss"]:.4f}')
    path = P.o('models', 'bert_lyrics_final'); model.save_pretrained(path); tok.save_pretrained(path)
    save_table(P, pd.DataFrame(hist), 'finetune_loss.csv')

def stage_embed(a, P):
    torch = need_torch()
    from transformers import BertTokenizer, BertModel
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); songs = load_songs(P)
    ft = a.bert_path or P.o('models', 'bert_lyrics_final')
    if not os.path.isdir(ft): raise SystemExit(f'fine-tuned model not found at {ft}: run "finetune" or pass --bert-path')
    def embed(path):
        tok = BertTokenizer.from_pretrained(path); enc = BertModel.from_pretrained(path).to(dev).eval(); c, m = [], []
        with torch.no_grad():
            for t in songs.text:
                x = tok(t, return_tensors='pt', truncation=True, padding=True, max_length=256).to(dev); h = enc(**x).last_hidden_state
                k = x['attention_mask'].unsqueeze(-1).float(); c.append(h[:, 0].cpu().numpy().ravel()); m.append(((h * k).sum(1) / k.sum(1)).cpu().numpy().ravel())
        return np.array(c), np.array(m)
    log('embedding with plain bert-base-uncased'); pc, pm = embed('bert-base-uncased')
    log(f'embedding with {ft}'); fc, fm = embed(ft)
    np.savez(P.o('embeddings.npz'), plain_cls=pc, plain_mean=pm, ft_cls=fc, ft_mean=fm); print('  -> embeddings.npz', fc.shape)


# ----------------------------------------------------------------------------- stage: song_nn  (neural models)
def build_models(torch):
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence
    def encode_music(lstm, seq):
        lengths = (seq[:, :, 0] != PAD).sum(dim=1).cpu()                      # true lengths -> reverse LSTM starts at the last real note
        seq = torch.where(seq == PAD, torch.zeros_like(seq), seq)
        _, (hn, _) = lstm(pack_padded_sequence(seq, lengths, batch_first=True, enforce_sorted=False))
        return torch.cat((hn[-2], hn[-1]), dim=1)
    class TextOnly(nn.Module):
        def __init__(s, d=768, k=6):
            super().__init__(); s.c = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, k), nn.Sigmoid())
        def forward(s, t, m=None): return s.c(t)
    class MusicOnly(nn.Module):
        def __init__(s, h=128, k=6):
            super().__init__(); s.lstm = nn.LSTM(1, h, num_layers=2, batch_first=True, bidirectional=True)
            s.c = nn.Sequential(nn.Linear(2 * h, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, k), nn.Sigmoid())
        def forward(s, t, m): return s.c(encode_music(s.lstm, m))
    class Fusion(nn.Module):
        def __init__(s, d=768, h=128, k=6):
            super().__init__(); s.lstm = nn.LSTM(1, h, num_layers=2, batch_first=True, bidirectional=True)
            s.tp, s.mp, s.att = nn.Linear(d, 256), nn.Linear(2 * h, 256), nn.Linear(512, 2)
            s.c = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, k), nn.Sigmoid())
        def forward(s, t, m, return_attention=False):
            tt, mm = torch.relu(s.tp(t)), torch.relu(s.mp(encode_music(s.lstm, m)))
            w = torch.softmax(s.att(torch.cat([tt, mm], 1)), dim=1); out = s.c(w[:, :1] * tt + w[:, 1:] * mm)
            return (out, w) if return_attention else out
    return {'text': TextOnly, 'music': MusicOnly, 'fusion': Fusion}

def stage_song_nn(a, P):
    from sklearn.model_selection import KFold
    songs = load_songs(P); Y = songs[EMO].values.astype(int); n = len(Y); seeds = [int(s) for s in a.seeds.split(',')]
    oof, att, fold_rows = {}, {}, []
    f_oof = P.o('song_nn_oof.npz')
    if a.reuse_oof:
        z = np.load(f_oof); log(f'reusing predictions in {f_oof}')
        for seed in seeds:
            for kind in ['text', 'music', 'fusion']: oof[(seed, kind)] = z[f'{seed}_{kind}']
            att[(seed, 'fusion')] = z[f'att_{seed}']
    else: oof, att = train_song_nn(a, P, songs, Y, seeds)
    for seed in seeds:
        for kind in ['text', 'music', 'fusion']:
            for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
                m = macro_micro_auc(Y, oof[(seed, kind)], THR_NN, te)
                fold_rows.append(dict(seed=seed, model=kind, fold=fold + 1, macro_f1=m['macro_f1'], micro_f1=m['micro_f1'], macro_auc=m['macro_auc'], **dict(zip(EMO, m['per_f1']))))
    song_nn_tables(P, Y, n, seeds, oof, att, fold_rows)

def train_song_nn(a, P, songs, Y, seeds):
    torch = need_torch(); import torch.nn as nn, torch.optim as optim
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.model_selection import KFold
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); emb = load_emb(P); n = len(Y)
    T = torch.FloatTensor(emb['ft_cls']); Yt = torch.FloatTensor(Y)
    M = pad_sequence([torch.FloatTensor(x).unsqueeze(1) for x in songs.intervals], batch_first=True, padding_value=PAD)
    log(f'music tensor {tuple(M.shape)} (longest sequence {M.shape[1]}); seeds {seeds}; device {dev}')
    classes = build_models(torch); oof, att = {}, {}
    for seed in seeds:
        for kind, cls in classes.items():
            P_ = np.zeros(Y.shape); W = np.zeros((n, 2))
            for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(np.arange(n))):
                torch.manual_seed(seed + fold); np.random.seed(seed + fold)
                model = cls().to(dev); opt = optim.Adam(model.parameters(), lr=1e-4); crit = nn.BCELoss()
                dl = DataLoader(TensorDataset(T[tr], M[tr], Yt[tr]), batch_size=16, shuffle=True)
                model.train()
                for _ in range(50):
                    for bt, bm, by in dl:
                        opt.zero_grad(); crit(model(bt.to(dev), bm.to(dev)), by.to(dev)).backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    if kind == 'fusion': o, w = model(T[te].to(dev), M[te].to(dev), return_attention=True); W[te] = w.cpu().numpy()
                    else: o = model(T[te].to(dev), M[te].to(dev))
                P_[te] = o.cpu().numpy()
            oof[(seed, kind)] = P_; att[(seed, kind)] = W; log(f'seed {seed} {kind}: pooled macro-F1 {macro_micro_auc(Y, P_, THR_NN)["macro_f1"]:.3f}')
            np.savez(P.o('song_nn_oof.npz'), **{f'{s}_{k}': v for (s, k), v in oof.items()}, **{f'att_{s}': v for (s, k), v in att.items() if k == 'fusion'})   # saved after every model
    return oof, att

def song_nn_tables(P, Y, n, seeds, oof, att, fold_rows):
    F = save_table(P, pd.DataFrame(fold_rows), 'song_nn_folds.csv')
    # ---- summaries: (1) thesis protocol (first seed): mean +- sd over folds; (2) across seeds
    names = {'text': 'Text-only', 'music': 'Music-only', 'fusion': 'Attention Fusion'}; cols = ['micro_f1', 'macro_f1'] + EMO; cols_a = cols + ['macro_auc']; rows, rows2 = [], []
    for k in ['text', 'music', 'fusion']:
        f0 = F[(F.model == k) & (F.seed == seeds[0])]
        rows.append({'Model': names[k], **{c: f'{f0[c].mean():.3f} ± {f0[c].std():.3f}' for c in cols_a}})
        per_seed = F[F.model == k].groupby('seed')[cols_a].mean()
        rows2.append({'Model': names[k], **{c: f'{per_seed[c].mean():.3f} ± {per_seed[c].std(ddof=0):.3f}' for c in cols_a}})
    save_table(P, pd.DataFrame(rows), 'song_nn_summary_first_seed.csv'); save_table(P, pd.DataFrame(rows2), 'song_nn_summary_across_seeds.csv')
    allpos = np.array([2 * Y[:, j].mean() / (1 + Y[:, j].mean()) for j in range(6)])
    save_table(P, pd.DataFrame([{'baseline': 'all-positive', 'macro_f1': allpos.mean(), 'micro_f1': 2 * Y.mean() / (1 + Y.mean()), **dict(zip(EMO, allpos))}]), 'song_nn_allpositive.csv')
    # ---- paired comparison: Attention Fusion - Text-only (per-seed differences + bootstrap over songs on seed-0 predictions)
    d = []; rb = boot_pair(Y, oof[(seeds[0], 'fusion')], oof[(seeds[0], 'text')], THR_NN, fold_ids(n, seeds[0]))
    for metric in ['macro_f1', 'macro_auc']:
        ps = [foldwise(Y, oof[(s, 'fusion')], THR_NN, fold_ids(n, s))[metric] - foldwise(Y, oof[(s, 'text')], THR_NN, fold_ids(n, s))[metric] for s in seeds]
        mu, lo, hi, sd = rb[metric]
        d.append(dict(metric=metric, mean_over_seeds=np.mean(ps), per_seed=' '.join(f'{x:+.3f}' for x in ps), seed0_boot_mean=mu, ci_low=lo, ci_high=hi, boot_se=sd))
    save_table(P, pd.DataFrame(d), 'song_nn_fusion_minus_text.csv')
    # ---- attention weights (fusion; seed-0 out-of-fold) and interval statistics
    W = att[(seeds[0], 'fusion')]
    att_rows = [{'emotion': e, 'n_songs': int(Y[:, j].sum()), 'mean_music_weight_if_positive': W[Y[:, j] == 1, 1].mean(),
                 'mean_music_weight_if_negative': W[Y[:, j] == 0, 1].mean()} for j, e in enumerate(EMO)]
    att_rows.append({'emotion': 'ALL songs', 'n_songs': n, 'mean_music_weight_if_positive': W[:, 1].mean(), 'mean_music_weight_if_negative': np.nan,
                     'sd_music_weight_all': W[:, 1].std(), 'max_music_weight_all': W[:, 1].max()})
    save_table(P, pd.DataFrame(att_rows), 'song_nn_attention.csv')
    # ---- threshold sensitivity (out-of-fold, fusion, seed-0) and figures
    rows = []
    for k in ['text', 'music', 'fusion']:
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
            m = macro_micro_auc(Y, oof[(seeds[0], k)], t); rows.append(dict(model=names[k], threshold=t, macro_f1=m['macro_f1'], micro_f1=m['micro_f1']))
    save_table(P, pd.DataFrame(rows), 'song_nn_threshold.csv')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4.5)); x = np.arange(len(cols)); w = 0.27
        for i, k in enumerate(['text', 'music', 'fusion']):
            f = F[(F.model == k) & (F.seed == seeds[0])]; ax.bar(x + (i - 1) * w, [f[c].mean() for c in cols], w, yerr=[f[c].std() for c in cols], capsize=3, label=names[k])
        ax.set_xticks(x); ax.set_xticklabels(['Micro-F1', 'Macro-F1'] + [e.capitalize() for e in EMO]); ax.set_ylabel('F1'); ax.legend(); ax.grid(axis='y', alpha=.4)
        ax.set_title(f'5-fold CV, seed {seeds[0]} (mean ± sd over folds)'); fig.tight_layout(); fig.savefig(P.o('fig_song_nn_f1.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for k in ['text', 'music', 'fusion']:
            r = [x for x in rows if x['model'] == names[k]]; ax.plot([x['threshold'] for x in r], [x['macro_f1'] for x in r], 'o-', label=names[k])
        ax.set_xlabel('decision threshold'); ax.set_ylabel('macro-F1 (out-of-fold)'); ax.legend(); ax.grid(alpha=.4); fig.tight_layout(); fig.savefig(P.o('fig_song_nn_threshold.png'), dpi=150); plt.close(fig)
    except Exception as e: print('  (figures skipped:', e, ')')


# ----------------------------------------------------------------------------- stage: song_linear (baselines, melody features, power)
def stage_song_linear(a, P):
    from scipy.stats import mannwhitneyu, ttest_ind
    from sklearn.model_selection import KFold
    from sklearn.metrics import f1_score
    songs = load_songs(P); emb = load_emb(P); Y = songs[EMO].values.astype(int); n = len(Y); T = emb['ft_cls']
    FS = {g: songs[['f_' + c for c in cols]].values for g, cols in GROUPS.items()}; ALLF = np.hstack(list(FS.values()))
    NEW = np.hstack([FS['rhythm/tempo'], FS['mode'], FS['pitch']])
    sets = {'text (LR)': T, 'melody: interval stats': FS['interval stats'], 'melody: rhythm/tempo': FS['rhythm/tempo'], 'melody: mode': FS['mode'],
            'melody: pitch': FS['pitch'], 'melody: all 24': ALLF, 'text + interval stats': np.hstack([T, FS['interval stats']]),
            'text + rhythm/mode/pitch': np.hstack([T, NEW]), 'text + all melody': np.hstack([T, ALLF])}
    R = 10; res = {k: [] for k in ['all-positive'] + list(sets)}; oofs = {}
    for s in range(R):
        for k, X in sets.items():
            Pk = oof_lr(X, Y, s)
            m = [macro_micro_auc(Y, Pk, 0.5, te) for _, te in KFold(5, shuffle=True, random_state=s).split(X)]    # per-fold, then averaged
            res[k].append(np.r_[np.mean([x['macro_f1'] for x in m]), np.mean([x['micro_f1'] for x in m]), np.mean([x['per_f1'] for x in m], 0), np.mean([x['macro_auc'] for x in m])])
            oofs[(s, k)] = Pk
        m = [macro_micro_auc(Y, np.ones(Y.shape), 0.5, te) for _, te in KFold(5, shuffle=True, random_state=s).split(T)]
        res['all-positive'].append(np.r_[np.mean([x['macro_f1'] for x in m]), np.mean([x['micro_f1'] for x in m]), np.mean([x['per_f1'] for x in m], 0), 0.5])
        log(f'repeat {s + 1}/{R} done')
    cols = ['macro_f1', 'micro_f1'] + EMO + ['macro_auc']
    tab = pd.DataFrame([{'model': k, **{c: np.mean(v, 0)[i] for i, c in enumerate(cols)}, 'macro_f1_sd_over_repeats': np.std(v, 0)[0]} for k, v in res.items()])
    save_table(P, tab.round(4), 'song_linear_results.csv')
    comps = []   # repeats are NOT independent samples (same 100 songs), so no p-value over repeats; uncertainty = bootstrap over songs on repeat-0 predictions
    for new in ['text + interval stats', 'text + rhythm/mode/pitch', 'text + all melody']:
        r = boot_pair(Y, oofs[(0, new)], oofs[(0, 'text (LR)')], 0.5, fold_ids(n, 0))
        for mi, mn in [(0, 'macro_f1'), (len(cols) - 1, 'macro_auc')]:
            d = np.array(res[new])[:, mi] - np.array(res['text (LR)'])[:, mi]
            comps.append(dict(comparison=f'{new} - text', metric=mn, mean_diff_over_repeats=d.mean(), repeats_positive=int((d > 0).sum()),
                              boot_ci_low=r[mn][1], boot_ci_high=r[mn][2]))
    save_table(P, pd.DataFrame(comps).round(4), 'song_linear_paired.csv')
    # ---- univariate tests, BH over all feature x emotion pairs
    rows = []
    for j, e in enumerate(EMO):
        for c in [c for g in GROUPS.values() for c in g]:
            x = songs['f_' + c]; pos, neg = x[Y[:, j] == 1], x[Y[:, j] == 0]
            rows.append(dict(emotion=e, feature=c, n_pos=int(len(pos)), p=mannwhitneyu(pos, neg).pvalue, median_diff=pos.median() - neg.median()))
    U = pd.DataFrame(rows); U['p_BH'] = bh(U.p.values); U = U.sort_values('p').round(5)
    save_table(P, U, 'song_linear_univariate.csv'); print(f'  BH p<0.05: {(U.p_BH < .05).sum()} of {len(U)} | raw p<0.05: {(U.p < .05).sum()} (chance ~{.05 * len(U):.1f})')
    # ---- power: (A) cluster-bootstrap SE of macro-F1 difference text+melody - text (5 seeds); (B) simulated single-feature power
    f1m = lambda Pk, i: np.mean([f1_score(Y[i, j], Pk[i, j] > 0.5, zero_division=0) for j in range(6)])
    ses, pts = [], []
    for s in range(5):
        a1, a0 = oofs[(s, 'text + rhythm/mode/pitch')], oofs[(s, 'text (LR)')]
        ses.append(boot_ci(lambda i: f1m(a1, i) - f1m(a0, i), n, seed=s)[3]); pts.append(f1m(a1, np.arange(n)) - f1m(a0, np.arange(n)))
    se = float(np.mean(ses)); rng = np.random.default_rng(1); sim = []
    for j, e in enumerate(EMO):
        y = np.r_[np.ones(int(Y[:, j].sum())), np.zeros(n - int(Y[:, j].sum()))].astype(bool)
        for dd in (0.3, 0.5, 0.8, 1.0):
            x = rng.standard_normal((2000, n)); x[:, y] += dd; p = ttest_ind(x[:, y], x[:, ~y], axis=1).pvalue
            sim.append(dict(emotion=e, n_pos=int(y.sum()), cohen_d=dd, power_alpha05=(p < .05).mean(), power_bonferroni=(p < .05 / len(U)).mean()))
    save_table(P, pd.DataFrame(sim).round(3), 'song_linear_power_single_feature.csv')
    save_table(P, pd.DataFrame([dict(observed_diff_macro_f1=np.mean(pts), bootstrap_se=se, min_detectable_effect_80pct=2.8 * se)]).round(4), 'song_linear_power_overall.csv')


# ----------------------------------------------------------------------------- stage: verse_level
def stage_verse_level(a, P):
    from scipy import sparse
    from scipy.stats import pearsonr
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    V = pd.read_pickle(P.o('verses.pkl')); g = V.song.values; S = V[EMO].values; Yb = (S >= 3).astype(int); n = len(V)
    def music(r):
        h = np.bincount(np.asarray(r.pitch) % 12, minlength=12) / len(r.pitch); return np.r_[h, np.mean(r.pitch), r.n_notes, r.is_minor]
    Xm = np.array([music(r) for r in V.itertuples()]); Xm[np.isnan(Xm[:, -1]), -1] = 0.5       # unknown key -> 0.5
    log(f'{n} lines; positives (score>=3): {dict(zip(EMO, Yb.sum(0)))}')
    def run(split, reg):
        oof = {k: np.zeros(S.shape) for k in ['text', 'music', 'both']}
        for tr, te in split:
            tf = TfidfVectorizer(min_df=2).fit(V.text.iloc[tr]); Tt, Te = tf.transform(V.text.iloc[tr]), tf.transform(V.text.iloc[te])
            sc = StandardScaler().fit(Xm[tr]); Mt, Me = sparse.csr_matrix(sc.transform(Xm[tr])), sparse.csr_matrix(sc.transform(Xm[te]))
            for k, (A, B) in {'text': (Tt, Te), 'music': (Mt, Me), 'both': (sparse.hstack([Tt, Mt]).tocsr(), sparse.hstack([Te, Me]).tocsr())}.items():
                if reg: oof[k][te] = Ridge(alpha=1.0).fit(A, S[tr]).predict(B)
                else:
                    for j in range(6): oof[k][te, j] = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000).fit(A, Yb[tr, j]).predict_proba(B)[:, 1]
        return oof
    from sklearn.metrics import roc_auc_score
    pr = lambda Pm, i: np.mean([pearsonr(Pm[i, j], S[i, j])[0] for j in range(6)])
    au = lambda Pm, i: np.mean([roc_auc_score(Yb[i, j], Pm[i, j]) for j in range(6)])
    splits = {'GroupKFold by song': lambda s: GroupKFold(10, shuffle=True, random_state=s).split(S, groups=g), 'plain KFold over lines (leaky)': lambda s: KFold(10, shuffle=True, random_state=s).split(S)}
    rows, keep = [], {}
    for task, reg, fn in [('regression (Pearson r)', True, pr), ('classification (macro-AUC, score>=3)', False, au)]:
        for sp, mk in splits.items():
            runs = [run(mk(s), reg) for s in range(3)]; keep[(task, sp)] = runs[0]
            for k in ['text', 'music', 'both']:
                rows.append(dict(task=task, split=sp, model={'both': 'text + music'}.get(k, k), value=np.mean([fn(o[k], np.arange(n)) for o in runs]),
                                 **{e: np.mean([(pearsonr(o[k][:, j], S[:, j])[0] if reg else roc_auc_score(Yb[:, j], o[k][:, j])) for o in runs]) for j, e in enumerate(EMO)}))
    save_table(P, pd.DataFrame(rows).round(4), 'verse_level_results.csv')
    songs = np.unique(g); by = {s: np.where(g == s)[0] for s in songs}; rng = np.random.default_rng(0); d = []
    for task, fn in [('regression (Pearson r)', pr), ('classification (macro-AUC, score>=3)', au)]:
        o = keep[(task, 'GroupKFold by song')]; v = []
        for _ in range(1000):
            idx = np.concatenate([by[s] for s in rng.choice(songs, len(songs))])
            try: v.append(fn(o['both'], idx) - fn(o['text'], idx))
            except ValueError: pass
        d.append(dict(task=task, diff_text_music_minus_text=np.mean(v), ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5)))
    save_table(P, pd.DataFrame(d).round(4), 'verse_level_bootstrap.csv')


# ----------------------------------------------------------------------------- stage: text_ablation
def stage_text_ablation(a, P):
    emb = load_emb(P); songs = load_songs(P); Y = songs[EMO].values.astype(int); n = len(Y)
    E = {'plain BERT (CLS)': emb['plain_cls'], 'plain BERT (mean-pool)': emb['plain_mean'], 'fine-tuned BERT (CLS)': emb['ft_cls'], 'fine-tuned BERT (mean-pool)': emb['ft_mean']}
    try: torch = need_torch(); import torch.nn as nn, torch.optim as optim; dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except SystemExit: torch = None; print('  (PyTorch missing: MLP part skipped)')
    from sklearn.model_selection import KFold
    def oof_mlp(X, seed):
        Pm = np.zeros(Y.shape); Xt, Yt = torch.FloatTensor(X), torch.FloatTensor(Y)
        for fold, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
            torch.manual_seed(seed * 100 + fold)
            m = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 6), nn.Sigmoid()).to(dev); opt = optim.Adam(m.parameters(), lr=1e-4)
            for _ in range(50):
                m.train(); perm = tr[torch.randperm(len(tr)).numpy()]
                for i in range(0, len(perm), 16):
                    b = perm[i:i + 16]; opt.zero_grad(); nn.BCELoss()(m(Xt[b].to(dev)), Yt[b].to(dev)).backward(); opt.step()
            m.eval()
            with torch.no_grad(): Pm[te] = m(Xt[te].to(dev)).cpu().numpy()
        return Pm
    clfs = [('LR', oof_lr, 0.5)] + ([('MLP', oof_mlp, THR_NN)] if torch else []); seeds = range(5); OOF = {}; rows = []
    for clf, fn, thr in clfs:
        for name, X in E.items():
            OOF[(clf, name)] = [fn(X, Y, s) if clf == 'LR' else fn(X, s) for s in seeds]; log(f'done {clf} {name}')
            m = [foldwise(Y, Pk, thr, fold_ids(n, s)) for s, Pk in zip(seeds, OOF[(clf, name)])]
            rows.append(dict(classifier=clf, embedding=name, **{e: np.mean([fold_auc_j(Y, Pk, j, fold_ids(n, s)) for s, Pk in zip(seeds, OOF[(clf, name)])]) for j, e in enumerate(EMO)},
                             macro_auc=np.mean([x['macro_auc'] for x in m]), sd_over_seeds=np.std([x['macro_auc'] for x in m]), macro_f1=np.mean([x['macro_f1'] for x in m])))
    save_table(P, pd.DataFrame(rows).round(4), 'text_ablation_table.csv')
    pairs = [('fine-tuned BERT (CLS)', 'plain BERT (CLS)'), ('fine-tuned BERT (CLS)', 'plain BERT (mean-pool)'), ('fine-tuned BERT (mean-pool)', 'plain BERT (mean-pool)')]; out = []
    for clf, fn, thr in clfs:
        for ft, pl in pairs:
            r = boot_pair(Y, OOF[(clf, ft)][0], OOF[(clf, pl)][0], thr, fold_ids(n, 0))
            for mt in ['macro_auc', 'macro_f1']:
                out.append(dict(classifier=clf, comparison=f'{ft} vs {pl}', metric=mt, difference=r[mt][0], ci_low=r[mt][1], ci_high=r[mt][2]))
    save_table(P, pd.DataFrame(out).round(4), 'text_ablation_diffs.csv')

def fold_auc_j(Y, Pk, j, fid):
    """AUC of emotion j computed inside each test fold, averaged over folds."""
    from sklearn.metrics import roc_auc_score
    v = [roc_auc_score(Y[fid == k, j], Pk[fid == k, j]) for k in np.unique(fid) if len(np.unique(Y[fid == k, j])) == 2]
    return np.mean(v) if v else np.nan


# ----------------------------------------------------------------------------- stage: summary
def stage_summary(a, P):
    order = ['finetune_loss', 'song_nn_summary_first_seed', 'song_nn_summary_across_seeds', 'song_nn_allpositive', 'song_nn_fusion_minus_text', 'song_nn_attention',
             'song_nn_threshold', 'song_linear_results', 'song_linear_paired', 'song_linear_univariate', 'song_linear_power_overall', 'song_linear_power_single_feature',
             'verse_level_results', 'verse_level_bootstrap', 'text_ablation_table', 'text_ablation_diffs']
    parts = ['# Results summary\n']
    for name in order:
        f = P.o(name + '.csv')
        if os.path.exists(f):
            df = pd.read_csv(f); df = df.head(20) if name == 'song_linear_univariate' else df
            parts.append(f'## {name}\n\n```\n{df.to_string(index=False)}\n```\n')
    open(P.o('SUMMARY.md'), 'w', encoding='utf-8').write('\n'.join(parts)); print('  -> ' + P.o('SUMMARY.md'))


STAGES = {'prepare': stage_prepare, 'finetune': stage_finetune, 'embed': stage_embed, 'song_nn': stage_song_nn, 'song_linear': stage_song_linear,
          'verse_level': stage_verse_level, 'text_ablation': stage_text_ablation, 'summary': stage_summary}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=['all'] + list(STAGES)); ap.add_argument('--data-dir', default='data'); ap.add_argument('--out-dir', default='results')
    ap.add_argument('--seeds', default='42,43,44,45,46', help='seeds for the neural song-level experiment; the first one follows the thesis protocol')
    ap.add_argument('--reuse-oof', action='store_true', help='song_nn: do not retrain, recompute all tables from an existing song_nn_oof.npz')
    ap.add_argument('--bert-path', default=None, help='use an existing fine-tuned model instead of results/models/bert_lyrics_final')
    ap.add_argument('--skip-finetune', action='store_true', help='with "all": reuse --bert-path / existing model')
    a = ap.parse_args(argv); P = Paths(a)
    for s in (list(STAGES) if a.stage == 'all' else [a.stage]):
        if s == 'finetune' and a.stage == 'all' and a.skip_finetune: continue
        log(f'=== stage: {s}'); STAGES[s](a, P)

if __name__ == '__main__':
    if 'ipykernel' in sys.modules:   # pasted into a notebook cell: sys.argv then belongs to the kernel, not to us
        print('Running inside a notebook cell. Either save this file and run it with\n'
              '    !python pipeline.py all --data-dir data --out-dir results\n'
              'or call main() with explicit arguments, e.g.\n'
              '    main(["prepare", "--data-dir", "data", "--out-dir", "results"])')
    else:
        main()
