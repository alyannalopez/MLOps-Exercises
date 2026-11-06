"""Step 8 delegation smoke test — imports brain + delegate DIRECTLY (no audio),
so it runs in any Python (the serve env lacks sounddevice)."""
import importlib.util
import sys
import time

SRC = "/Users/yannamii/sandbox/ME3/src"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load brain WITHOUT triggering voiceduplex.loop.__init__ (which imports audioio
# -> sounddevice). brain.py only needs smarthome (pure python).
import yaml

# Register the parent packages as namespace-ish so brain's relative imports resolve.
import types
vd = types.ModuleType("voiceduplex"); vd.__path__ = [SRC + "/voiceduplex"]; sys.modules["voiceduplex"] = vd
loop = types.ModuleType("voiceduplex.loop"); loop.__path__ = [SRC + "/voiceduplex/loop"]; sys.modules["voiceduplex.loop"] = loop
sh = types.ModuleType("voiceduplex.smarthome"); sh.__path__ = [SRC + "/voiceduplex/smarthome"]; sys.modules["voiceduplex.smarthome"] = sh
dl = types.ModuleType("voiceduplex.delegate"); dl.__path__ = [SRC + "/voiceduplex/delegate"]; sys.modules["voiceduplex.delegate"] = dl

# smarthome submodules brain imports
for sub in ("device", "weather"):
    load(f"voiceduplex.smarthome.{sub}", f"{SRC}/voiceduplex/smarthome/{sub}.py")

brain = load("voiceduplex.loop.brain", f"{SRC}/voiceduplex/loop/brain.py")
delegate_pkg = load("voiceduplex.delegate", f"{SRC}/voiceduplex/delegate/__init__.py")

cfg = yaml.safe_load(open("/Users/yannamii/sandbox/ME3/config/config.yaml"))
backend = delegate_pkg.build_backend(cfg)
print("backend:", backend.name, "available:", backend.is_available(), flush=True)

d = brain.route("what time is it", delegate=backend)
print("quick  ->", d.route, "|", d.reply, "| pending=", d.pending, flush=True)

d = brain.route("mm-hm", delegate=backend)
print("backch ->", d.route, "|", d.reply, flush=True)

t0 = time.monotonic()
d = brain.route("write a haiku about the sea and save it to haiku.txt", delegate=backend)
print("delegate->", d.route, "| ack=", d.reply, "| pending?", d.pending is not None, flush=True)
print("  ...waiting for OnIt...", flush=True)
res = d.pending.wait(140)
print("  RESULT ok=%s elapsed=%.1fs cancelled=%s" % (res.ok, res.elapsed, res.cancelled), flush=True)
print("  ANSWER:", (res.text[:400] if res.ok else res.error), flush=True)
print("total wall: %.1fs" % (time.monotonic() - t0), flush=True)
