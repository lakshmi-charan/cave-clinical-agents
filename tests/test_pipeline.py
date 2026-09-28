"""End-to-end pipeline test with a mock FHIR server and scripted dummy models."""
import hashlib, json, os, re, shutil, subprocess, sys, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CAVE_CONFIG"] = str(ROOT / "tests" / "config_test.yaml")
shutil.rmtree(ROOT / "runs_test", ignore_errors=True)
srv = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_fhir.py"), str(ROOT / "cave" / "test_data_v2.json"), "18080"])
time.sleep(1.5)

from cave import llm


def h(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def hook(model, system, messages):
    first = messages[0]["content"]
    rnd = h(model + json.dumps(messages)) % 1000 / 1000
    if system:  # verifier
        wrongish = "4 g" in first or '"value": 4' in first
        p = 0.75 if not wrongish else 0.6
        dec = "approve" if rnd < p else "reject"
        return json.dumps({"decision": dec, "confidence": int(50 + 50 * rnd), "justification": f"checked {dec} {rnd:.2f}"})
    q = first.split("Question:")[-1]
    ctx = first.split("Context:")[-1]
    mrn = re.search(r"S\d{7}", q)
    code = re.search(r'code for \w+ is \\?"(\w+)', ctx)
    n_asst = sum(1 for m in messages if m["role"] == "assistant")
    api = re.search(r"(http://\S+?/fhir/)", first).group(1)
    if n_asst == 0:
        if "$lab-trend" in first and mrn and code:
            return f"THOUGHT: use trend tool\nGET {api}$lab-trend?subject={mrn.group(0)}&code={code.group(1)}"
        if "LabResults" in first and mrn and code:
            return f"THOUGHT: aliased lab tool\nGET {api}LabResults?analyte={code.group(1)}&subjectId={mrn.group(0)}"
        if mrn and code:
            return f"THOUGHT: look up the lab\nGET {api}Observation?patient={mrn.group(0)}&code={code.group(1)}"
        m = re.search(r"named (\w+) (\w+)", q) or re.search(r"name (\w+) (\w+)", q)
        if m:
            return f"THOUGHT: find patient\nGET {api}Patient?given={m.group(1)}&family={m.group(2)}"
        if mrn:
            return f"THOUGHT: read patient\nGET {api}Patient?_id={mrn.group(0)}"
        return "THOUGHT: nothing\nFINISH([-1])"
    last = messages[-1]["content"]
    vals = re.findall(r'"value":([0-9.]+)', last) or re.findall(r'\[\d+,([0-9.]+)\]', last)
    if "order" in q.lower() and n_asst == 1 and rnd < 0.7:
        ndc = "0338-1715-40" if "magnesium" in q.lower() else "40032-917-01"
        dose = [1, 2, 4][h(first) % 3] if ndc.startswith("0338") else [10, 20, 30, 40][h(first) % 4]
        unit = "g" if ndc.startswith("0338") else "mEq"
        pid = mrn.group(0) if mrn else (re.findall(r'"id":"(S\d+)"', last) or ["S0000000"])[0]
        body = {"resourceType": "MedicationRequest", "medicationCodeableConcept": {"coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": ndc}]},
                "authoredOn": "2023-11-13T10:15:00+00:00", "dosageInstruction": [{"route": {"text": "IV"}, "doseAndRate": [{"doseQuantity": {"value": dose, "unit": unit}}]}],
                "status": "active", "intent": "order", "subject": {"reference": f"Patient/{pid}"}}
        return "THOUGHT: level is low so order replacement\nPOST " + api + "MedicationRequest\n" + json.dumps(body)
    v = vals[-1] if vals and rnd < 0.8 else "-1"
    return f"THOUGHT: report value\nFINISH([{v}])"


llm.set_dummy_hook(hook)
import run
A = types.SimpleNamespace(n=2, limit=0, splits=None, boot=50)
try:
    run.cmd_check(A) if False else None
    run.cmd_base(A)
    run.cmd_stress_setup(A)
    run.cmd_stress(A)
    run.cmd_verify(A)
    run.cmd_analyze(A)
finally:
    srv.terminate()
