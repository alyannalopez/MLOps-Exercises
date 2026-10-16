import json
d = json.load(open("models/crnn_seed0/train_log.json"))
print("TYPE:", type(d).__name__)
if isinstance(d, dict):
    print("KEYS:", list(d.keys()))
    for k, v in d.items():
        if isinstance(v, list):
            print(f"  {k}: list[{len(v)}], sample[0]={v[0] if v else None}")
        else:
            print(f"  {k}: {v!r}")
elif isinstance(d, list):
    print("LEN:", len(d))
    print("FIRST:", d[0])
    print("LAST:", d[-1])
