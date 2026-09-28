"""Tiny in-memory FHIR server imitating the MedAgentBench HAPI server (for offline pipeline tests only)."""
import json, random, re, sys, threading, uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode

NOW = datetime(2023, 11, 13, 10, 15, tzinfo=timezone.utc)
DB = {}  # resourceType -> {id: resource}
LOCK = threading.Lock()


def seed(data_path):
    rng = random.Random(1)
    data = json.load(open(data_path))
    ids = set(d["eval_MRN"] for d in data if "eval_MRN" in d)
    names = {}
    for d in data:
        if d["id"].startswith("task1_") and d["sol"][0] != "Patient not found":
            m = re.search(r"name (\S+) (\S+) and DOB of (\d{4}-\d{2}-\d{2})", d["instruction"])
            names[d["sol"][0]] = (m.group(1), m.group(2), m.group(3))
            ids.add(d["sol"][0])
    while len(ids) < 110:
        ids.add(f"S{rng.randint(1000000, 6999999)}")
    DB["Patient"] = {}
    DB["Observation"] = {}
    for pid in sorted(ids):
        g, f, b = names.get(pid, (rng.choice(["Ann", "Bob", "Cara", "Dev", "Eli"]), rng.choice(["Stone", "Reyes", "Kim", "Olsen"]),
                                 f"{rng.randint(1930, 2000)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"))
        DB["Patient"][pid] = {"resourceType": "Patient", "id": pid, "meta": {"versionId": "1"},
                              "identifier": [{"system": "urn:mrn", "value": pid}],
                              "name": [{"use": "official", "family": f, "given": [g]}], "gender": "female", "birthDate": b}
        for code, lo, hi, unit, n in [("MG", 0.8, 2.6, "mg/dL", 4), ("K", 2.7, 5.0, "mmol/L", 4), ("GLU", 60, 300, "mg/dL", 6), ("A1C", 5.0, 10.0, "%", 2)]:
            for i in range(rng.randint(0, n)):
                h = rng.choice([rng.uniform(1, 23), rng.uniform(24, 24 * 700)])
                oid = str(uuid.UUID(int=rng.getrandbits(128)))[:8]
                DB["Observation"][oid] = {"resourceType": "Observation", "id": oid, "meta": {"versionId": "1"}, "status": "final",
                                          "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
                                          "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": code}], "text": code},
                                          "subject": {"reference": f"Patient/{pid}"},
                                          "effectiveDateTime": (NOW - timedelta(hours=h)).isoformat(),
                                          "valueQuantity": {"value": round(rng.uniform(lo, hi), 1), "unit": unit}}
    for r in ("MedicationRequest", "AllergyIntolerance", "ServiceRequest", "Condition", "Procedure"):
        DB[r] = {}


def _pd(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if len(s) > 10 else datetime.fromisoformat(s + "T00:00:00+00:00")


def match(res, params):
    for k, v in params.items():
        if k in ("_count", "_format", "_offset"):
            continue
        if k in ("patient", "subject"):
            ref = (res.get("subject") or res.get("patient") or {}).get("reference", "")
            if ref.split("/")[-1] != v.split("/")[-1]:
                return False
        elif k == "code":
            codes = [c.get("code") for c in (res.get("code") or {}).get("coding", [])]
            if v not in codes:
                return False
        elif k == "category":
            cats = [c.get("code") for cat in res.get("category", []) for c in cat.get("coding", [])] if isinstance(res.get("category"), list) else []
            if v not in cats:
                return False
        elif k == "date":
            m = re.match(r"(ge|le|gt|lt|eq)?(.*)", v)
            op, val = m.group(1) or "eq", _pd(m.group(2))
            t = res.get("effectiveDateTime")
            if not t:
                return False
            t = _pd(t)
            if op == "ge" and t < val or op == "le" and t > val or op == "gt" and t <= val or op == "lt" and t >= val:
                return False
        elif k == "given":
            if not any(v.lower() in [g.lower() for g in n.get("given", [])] for n in res.get("name", [])):
                return False
        elif k == "family":
            if not any(v.lower() == n.get("family", "").lower() for n in res.get("name", [])):
                return False
        elif k == "birthdate":
            if res.get("birthDate") != v:
                return False
        elif k in ("_id", "identifier"):
            if res.get("id") != v.split("|")[-1]:
                return False
        elif k == "name":
            if not any(v.lower() in json.dumps(n).lower() for n in res.get("name", [])):
                return False
        else:
            raise ValueError(f"unknown search parameter {k}")
    return True


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/fhir+json;charset=utf-8" if code < 400 else "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p][1:]
        params = dict(parse_qsl(u.query))
        if parts == ["metadata"]:
            return self._send(200, {"resourceType": "CapabilityStatement"})
        if not parts or parts[0] not in DB:
            return self._send(404, {"error": "unknown resource"})
        if len(parts) == 2:
            r = DB[parts[0]].get(parts[1])
            return self._send(200, r) if r else self._send(404, {"error": "not found"})
        try:
            hits = [r for r in DB[parts[0]].values() if match(r, params)]
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        cnt, off = int(params.get("_count", 20)), int(params.get("_offset", 0))
        page = hits[off:off + cnt]
        b = {"resourceType": "Bundle", "type": "searchset", "total": len(hits), "meta": {"lastUpdated": "x"},
             "link": [{"relation": "self", "url": self.path}],
             "entry": [{"fullUrl": f"http://localhost/fhir/{parts[0]}/{r['id']}", "resource": r, "search": {"mode": "match"}} for r in page]}
        if off + cnt < len(hits):
            p2 = dict(params, _offset=off + cnt, _count=cnt)
            b["link"].append({"relation": "next", "url": f"http://{self.headers['Host']}{u.path}?{urlencode(p2)}"})
        self._send(200, b)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n))

    def do_POST(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p][1:]
        r = self._body()
        with LOCK:
            rid = str(len(DB[parts[0]]) + 100000)
            r["id"] = rid
            DB[parts[0]][rid] = r
        self._send(201, r)

    def do_PUT(self):
        parts = [p for p in urlparse(self.path).path.split("/") if p][1:]
        r = self._body()
        r["id"] = parts[1]
        DB[parts[0]][parts[1]] = r
        self._send(201, r)


if __name__ == "__main__":
    seed(sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[2])), H).serve_forever()
