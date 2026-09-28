"""Re-check stress-task reference solutions against the current EHR (no writes). Prints IDENTICAL or the diffs."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cave.common import load_env, out_dir, read_json
from cave.reference import solve
load_env()
old = read_json(out_dir("tasks") / "stress_tasks.json")
bad = []
for t in old:
    new = solve(t)
    if new != t["ref"]:
        bad.append((t["id"], t["ref"], new))
if not bad:
    print("IDENTICAL: all", len(old), "stress references match")
else:
    for tid, a, b in bad:
        print("MISMATCH", tid, "\n  stored:", json.dumps(a)[:400], "\n  now:   ", json.dumps(b)[:400])
