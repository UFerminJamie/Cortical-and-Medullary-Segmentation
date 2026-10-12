"""Check a running CPU container against generated synthetic data (stdlib only)."""
import argparse
import http.cookiejar
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    for _ in range(120):
        try:
            urllib.request.urlopen(base+"/health", timeout=2)
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("Container did not become healthy within 120 seconds")
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    token = (args.state/"access_token").read_text().strip()
    opener.open(base+"/?token="+token, timeout=10)

    def get(path):
        return json.load(opener.open(base+path, timeout=10))

    assert get("/api/bootstrap")["compute_enabled"] is False
    rows = get("/api/slides")
    assert len(rows) == 4 and all(row["name"].startswith("DEMO_") for row in rows), "Only the synthetic demo is supported"
    sid = rows[0]["id"]
    detail = get("/api/slides/"+sid)
    assert opener.open(base+f"/api/wsi/{sid}_files/12/0_0.jpg").read().startswith(b"\xff\xd8")
    assert len(opener.open(base+f"/api/slides/{sid}/view").read()) > 100
    if not args.check_existing:
        payload = {"label":1, "polygon":[[650,650],[1150,650],[1150,1150],[650,1150]],
                   "revision":detail["revision"], "fingerprint":detail["fingerprint"],
                   "reviewer":"Synthetic container check", "pass_id":"pass-1",
                   "visible_only":False, "grid_format":2}
        request = urllib.request.Request(base+f"/api/slides/{sid}/corrections", data=json.dumps(payload).encode(),
                                         headers={"Content-Type":"application/json"}, method="POST")
        assert len(json.load(opener.open(request))["tile_ids"]) == 16
    assert get("/api/slides/"+sid)["manual_tiles"] == 16
    print("CPU container check passed: authenticated API, data mount, tiles, overlays, persistent annotations.")


if __name__ == "__main__":
    main()
