"""Neural listwise ranker per field (PyTorch, GPU). Encoder maps each candidate's
pair-features to a score; listwise softmax cross-entropy to the true candidate.
Trained with shift augmentation. Evaluated on clean-CV and shift-CV; OOF scores
saved for ensembling with the GBDT models.

Small data (600 rows) -> compact net, dropout, weight decay, early-ish stopping.
"""
import sys, json, csv, time, argparse
import numpy as np
sys.path.insert(0, "research"); sys.path.insert(0, ".")
from featx import build_row_field, FIELDS, OPTIONS_KEY
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
csv.field_size_limit(10**8)

CFG = dict(raw_diff=1, absdist=1, cos=1, absval=1, vendor=1, hint=1, rel=1, comp=1)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


def load_rows(path):
    out = []
    for row in csv.DictReader(open(path, encoding="utf-8")):
        rec = {"id": row["id"], "corrupted_card": json.loads(row["corrupted_card"]),
               "support_cards": json.loads(row["support_cards"]),
               "source_options": json.loads(row["source_options"]),
               "name_type_options": json.loads(row["name_type_options"]),
               "library_options": json.loads(row["library_options"])}
        if row.get("answer_json"): rec["answer"] = json.loads(row["answer_json"])
        out.append(rec)
    return out


def build_field(rows, field, perturb_seed=None):
    out = []
    for ri, r in enumerate(rows):
        rng = np.random.default_rng(perturb_seed + ri) if perturb_seed is not None else None
        X, toks, names, isc = build_row_field(r["corrupted_card"], r["support_cards"],
                                              r[OPTIONS_KEY[field]], field, CFG, rng)
        truth = r["answer"][field]
        yi = [i for i, t in enumerate(toks) if t == truth]
        out.append((X.astype(np.float32), toks, truth, isc, yi[0] if yi else -1))
    return out


class Net(nn.Module):
    def __init__(self, d, h=128, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p),
            nn.Linear(h, 1))
    def forward(self, x):  # x: [N, d] -> [N]
        return self.net(x).squeeze(-1)


def train_fold(train_data, mean, std, d, epochs=60, lr=1e-3, wd=1e-4, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    net = Net(d).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        net.train()
        order = np.random.permutation(len(train_data))
        for ri in order:
            X, yi = train_data[ri]
            if yi < 0:
                continue
            xt = torch.from_numpy((X - mean)/std).to(DEV)
            logits = net(xt).unsqueeze(0)              # [1, N]
            loss = ce(logits, torch.tensor([yi], device=DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return net


def run(rows, n_aug, perturb_val, epochs):
    n = len(rows)
    kf = KFold(5, shuffle=True, random_state=SEED)
    field_correct = {}
    field_oof = {}
    for field in FIELDS:
        clean = build_field(rows, field, None)
        augs = [build_field(rows, field, 5000 + 777*k) for k in range(n_aug)]
        valp = build_field(rows, field, 9000) if perturb_val else None
        d = clean[0][0].shape[1]
        correct = np.zeros(n, bool)
        oof_scores = [None]*n
        for tr, va in kf.split(np.arange(n)):
            # normalization stats from clean train rows
            allX = np.vstack([clean[ri][0] for ri in tr])
            mean = allX.mean(0, keepdims=True); std = allX.std(0, keepdims=True) + 1e-6
            train_data = []
            for ri in tr:
                for ds in [clean] + augs:
                    train_data.append((ds[ri][0], ds[ri][4]))
            net = train_fold(train_data, mean, std, d, epochs=epochs)
            net.eval()
            with torch.no_grad():
                for ri in va:
                    ds = valp if perturb_val else clean
                    X, toks, truth, isc, yi = ds[ri]
                    xt = torch.from_numpy((X - mean)/std).to(DEV)
                    sc = net(xt).cpu().numpy().copy()
                    oof_scores[ri] = sc.copy()
                    sc[isc.astype(bool)] = -1e9
                    correct[ri] = (toks[int(np.argmax(sc))] == truth)
        field_correct[field] = correct
        field_oof[field] = oof_scores
        print(f"  [{'shift' if perturb_val else 'clean'}] neural {field:16s} acc={correct.mean():.3f}", flush=True)
    at = np.ones(n, bool)
    for f in FIELDS: at &= field_correct[f]
    fa = np.mean([field_correct[f].mean() for f in FIELDS])
    print(f"  [{'shift' if perturb_val else 'clean'}] neural ALL3={at.mean():.3f} score={0.97*at.mean()+0.03*fa:.4f}", flush=True)
    return field_oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_aug", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    rows = load_rows("dataset/public/train.csv")
    print(f"rows={len(rows)} dev={DEV} n_aug={args.n_aug} epochs={args.epochs}", flush=True)
    t0 = time.time()
    for pv in [False, True]:
        oof = run(rows, args.n_aug, pv, args.epochs)
        if not pv:
            np.savez("research/neural_oof_clean.npz",
                     **{f: np.array(oof[f], dtype=object) for f in FIELDS})
    print(f"elapsed {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
