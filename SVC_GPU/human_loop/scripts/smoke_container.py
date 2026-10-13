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

    def write(path, payload, method='POST'):
        request = urllib.request.Request(base+path, data=json.dumps(payload).encode(),
                                         headers={'Content-Type':'application/json'}, method=method)
        return json.load(opener.open(request, timeout=20))

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
    layer_url = f'/api/slides/{sid}/annotations/cortical_gt.json'
    if not args.check_existing:
        created = write(layer_url, {'reviewer':'Container check', 'revision':None, 'create':True}, 'PUT')
        polygon = {'type':'Polygon', 'coordinates':[[[650,650],[1150,650],[1150,1150],[650,1150],[650,650]]]}
        write(layer_url, {'reviewer':'Container check', 'revision':created['revision'],
                         'operations':[{'action':'add','geometry':polygon}]}, 'PUT')
        tissue = {'type':'FeatureCollection','features':[{'type':'Feature','properties':{},'geometry':{
            'type':'Polygon','coordinates':[[[0,0],[detail['width'],0],[detail['width'],detail['height']],[0,detail['height']],[0,0]]]}}]}
        write(f'/api/slides/{sid}/annotations/Tissue_qc.json', {'reviewer':'Container check','revision':None,'document':tissue}, 'PUT')
        generation = write(f'/api/slides/{sid}/annotation-generation', {'reviewer':'Container check'})
        assert generation['status'] == 'current'
    assert len(get(layer_url)['regions']) == 1
    assert len(get(f'/api/slides/{sid}/annotations/cortical.json')['regions']) == 1
    assert get(f'/api/slides/{sid}/annotations')['generation']['status'] == 'current'
    print("CPU container check passed: authenticated API, data mount, tiles, overlays, persistent annotations.")


if __name__ == "__main__":
    main()
