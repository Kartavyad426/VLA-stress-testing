"""Category overlap by head noun: LIBERO-Plus new_objects categories vs Kitchen-MJCF categories vs GSO model names."""
import sys, os, glob, re, json, collections
LP, K, G, out = sys.argv[1:5]
def toks(s): return [t for t in re.split(r"[_\W]+", s.lower()) if t]
def head(cat):
    t = toks(cat)
    if "of" in t: t = t[:t.index("of")]   # "bottle_of_beer" -> bottle
    return t[-1] if t else cat
def sing(w): return w[:-1] if w.endswith("s") and not w.endswith("ss") and len(w) > 3 else w
lp = sorted(os.listdir(LP))
kc = sorted(os.listdir(K))
kmodels = {c: len(os.listdir(f"{K}/{c}")) for c in kc}
gso = sorted(os.listdir(G))
gso_tok = {g: {sing(t) for t in toks(g)} for g in gso}
res = {}
for c in lp:
    h = sing(head(c))
    kh = [k for k in kc if sing(head(k)) == h or k == c]
    gh = [g for g in gso if h in gso_tok[g]]
    res[c] = dict(head=h, kitchen=kh, gso=gh)
k_new = [k for k in kc if not any(sing(head(k)) == r["head"] for r in res.values())]
summary = dict(lp_categories=len(lp), lp_with_kitchen=sum(bool(r["kitchen"]) for r in res.values()),
               lp_with_gso=sum(bool(r["gso"]) for r in res.values()),
               lp_with_either=sum(bool(r["kitchen"] or r["gso"]) for r in res.values()),
               kitchen_categories=len(kc), kitchen_models=sum(kmodels.values()), kitchen_not_in_lp=k_new)
json.dump(dict(summary=summary, per_category=res), open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
for c in ["apple", "half_radish", "radish", "bowl"]:
    for k in res:
        if k == c or k.endswith("_" + c): print(k, res[k]["kitchen"], res[k]["gso"][:6])
