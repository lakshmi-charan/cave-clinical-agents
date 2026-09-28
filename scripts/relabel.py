"""Semantic (clinical-content) relabeling of logged actions, alongside the strict-format labels.

Strict: the answer must be in the exact JSON format the grader expects (as in the official benchmark).
Semantic: the clinically relevant content must be right (value, patient, order), whatever the wording.
Usage:  python scripts/relabel.py AGENT_RUN
Output: runs/results/labels_semantic_<run>.csv and runs/results/tasks_semantic_<run>.csv
"""
import csv, glob, json, re, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cave.reference import order_matches, answer_ok, parse_answer, to_float

AGENT = sys.argv[1] if len(sys.argv) > 1 else "gpt-5-mini"
tasks = {t["id"]: t for t in json.load(open(f"{ROOT}/runs/tasks/base_tasks.json", encoding="utf-8"))}
sp = f"{ROOT}/runs/tasks/stress_tasks.json"
if os.path.exists(sp):
    tasks.update({t["id"]: t for t in json.load(open(sp, encoding="utf-8"))})

NUM = re.compile(r"(?<![\w\-./:])(-?\d+(?:\.\d+)?)(?![\w\-/:]|\.\d)")
DATE = re.compile(r"\d{4}-\d{2}-\d{2}(?:T[\d:.+\-Z]+)?")
MRN = re.compile(r"\bS\d{7}\b")
NEG = re.compile(r"\b(no|not|none|unavailable|without)\b", re.I)


def text_of(raw):
    ans = parse_answer(raw)
    return " ; ".join(str(a) for a in ans), ans


def numbers(txt):
    t = DATE.sub(" ", txt)
    t = MRN.sub(" ", t)
    t = re.sub(r"\bid[: ]+\w+", " ", t, flags=re.I)
    return [float(x) for x in NUM.findall(t)]


def sem_answer_ok(spec, raw):
    if spec is None:
        return True
    if answer_ok(spec, raw):
        return True
    txt, ans = text_of(raw)
    k = spec["kind"]
    if k == "str":
        v = str(spec["value"])
        if v == "Patient not found":
            return bool(re.search(r"not found|no (matching )?patient|does not exist", txt, re.I)) and not MRN.search(txt)
        return MRN.findall(txt)[:1] == [v]
    nums = numbers(txt)
    if k == "int":
        return bool(nums) and nums[0] == spec["value"]
    if k == "num":
        if spec["value"] == -1:
            return (bool(nums) and nums[0] == -1) or (not nums and bool(NEG.search(txt)))
        return bool(nums) and abs(nums[0] - spec["value"]) <= spec["tol"]
    if k == "a1c":
        if spec["value"] == -1:
            return (bool(nums) and nums[0] == -1) or (not nums and bool(NEG.search(txt)))
        return bool(nums) and abs(nums[0] - spec["value"]) <= 0.05 and str(spec["date"])[:10] in txt
    return False


def loose(s):
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def sem_order_matches(spec, resource, payload):
    if order_matches(spec, resource, payload):
        return True
    if spec.get("note") and resource == "ServiceRequest":
        s2 = dict(spec); note = s2.pop("note")
        if not order_matches(s2, resource, payload):
            return False
        notes = payload.get("note")
        txt = " ".join(n.get("text", "") for n in notes) if isinstance(notes, list) else \
            (notes.get("text", "") if isinstance(notes, dict) else str(notes or ""))
        return loose(note) in loose(txt)
    return False


def label(ref, units, sem):
    om = sem_order_matches if sem else order_matches
    ao = sem_answer_ok if sem else answer_ok
    pending, optional = list(ref["orders"]), list(ref["optional_orders"])
    out, all_ok = [], True
    for u in units:
        if u["type"] == "POST":
            hit = None
            for lst, tag in ((pending, "req"), (optional, "opt")):
                for i, s in enumerate(lst):
                    if om(s, u["resource"], u["payload"]):
                        hit = (lst, i); break
                if hit: break
            if hit:
                hit[0].pop(hit[1]); out.append(1)
            else:
                out.append(0); all_ok = False
        else:
            out.append(None)
    fin_ok = None
    for j, u in enumerate(units):
        if u["type"] == "FINISH":
            fin_ok = int(ao(ref["answer"], u["answer_raw"]) and not pending)
            out[j] = fin_ok
    return out, int(bool(fin_ok) and all_ok)


rows, trows = [], []
for f in sorted(glob.glob(f"{ROOT}/runs/agents/{AGENT}/*.json")):
    r = json.load(open(f, encoding="utf-8"))
    t = tasks[r["task_id"]]
    strict, s_ok = label(t["ref"], r["units"], False)
    sem, m_ok = label(t["ref"], r["units"], True)
    for i, u in enumerate(r["units"]):
        rows.append({"task_id": r["task_id"], "unit": i, "type": u["type"], "correct_strict": strict[i],
                     "correct_sem": sem[i], "correct_logged": int(bool(u["correct"]))})
    trows.append({"task_id": r["task_id"], "success_strict": s_ok, "success_sem": m_ok,
                  "success_logged": int(r["success_internal"])})
os.makedirs(f"{ROOT}/runs/results", exist_ok=True)
for name, data in ((f"labels_semantic_{AGENT}.csv", rows), (f"tasks_semantic_{AGENT}.csv", trows)):
    with open(f"{ROOT}/runs/results/{name}", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print("units", len(rows), "strict==logged", sum(r["correct_strict"] == r["correct_logged"] for r in rows),
      "sem correct", sum(r["correct_sem"] for r in rows), "strict correct", sum(r["correct_strict"] for r in rows))
print("tasks", len(trows), "strict", sum(t["success_strict"] for t in trows), "sem", sum(t["success_sem"] for t in trows))
