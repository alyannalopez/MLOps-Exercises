"""ME2 VCM runtime — real-time voice-controlled smart device.

Pipeline:  mic (16 kHz) -> openWakeWord ("hey mycroft") -> on trigger, record
the command (up to 2 s, energy-based endpointing) -> conformer_s ONNX
classifier -> skill dispatcher -> simulated smart home -> spoken/text
response.

Modes:
  python -m runtime.main --mic                     live microphone
  python -m runtime.main --file clip.wav           replay one clip
  python -m runtime.main --demo                    end-to-end demo over the
                                                   held-out test set (no mic)

Everything runs locally: time/date from the system clock, weather from the
offline simulator, all actuators simulated.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.device import SimulatedSmartHome  # noqa: E402
from runtime.skills import CATEGORY_BY_INTENT, SkillDispatcher  # noqa: E402
from runtime.vcm_runtime import SAMPLE_RATE, VCMClassifier  # noqa: E402

log = logging.getLogger("me2.runtime")

DEFAULT_ONNX = ROOT / "models" / "conformer_s_float.onnx"
COMMAND_WINDOW_S = 2.0          # max command recording after wake word
ENDPOINT_SILENCE_S = 0.45       # stop recording after this much trailing silence
ENDPOINT_RMS = 0.010            # silence threshold (float32 rms)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def record_command(stream, max_s: float = COMMAND_WINDOW_S) -> np.ndarray:
    """Record from an active sounddevice stream until trailing silence."""
    frame = int(SAMPLE_RATE * 0.02)  # 20 ms
    buf: list[np.ndarray] = []
    silence_start: float | None = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_s:
        data, _ = stream.read(frame)
        seg = np.asarray(data, dtype=np.float32).flatten()
        buf.append(seg)
        if _rms(seg) < ENDPOINT_RMS:
            silence_start = silence_start or time.monotonic()
        else:
            silence_start = None
        if silence_start is not None and (time.monotonic() - silence_start) >= ENDPOINT_SILENCE_S:
            break
    audio = np.concatenate(buf) if buf else np.zeros(SAMPLE_RATE, np.float32)
    # keep the loudest 1-second window (matches the training-time crop)
    n = len(audio)
    if n > SAMPLE_RATE:
        win = SAMPLE_RATE // 2
        step = 160
        energies = [_rms(audio[i:i + win]) for i in range(0, n - win, step)]
        best = int(np.argmax(energies)) * step
        audio = audio[best:best + SAMPLE_RATE]
    return audio


def handle_command(dispatcher: SkillDispatcher, classifier: VCMClassifier,
                   audio: np.ndarray, phrase: str = "",
                   gate: bool = True) -> dict:
    res = classifier.classify_gated(audio) if gate else classifier.classify(audio)
    if res is None:
        return {"acted": False, "reason": "below_confidence_gate", "phrase": phrase}
    # Slot comes directly from the two-head model (no ASR).
    # For slotted intents, res.slot is the model's prediction (e.g. "10 seconds").
    # Fall back to phrase parsing only if the model didn't produce a slot.
    slot = res.slot
    if not slot and phrase:
        slot = None  # phrase is kept as a fallback for the dispatcher
    outcome = dispatcher.dispatch(res.intent, phrase=phrase, slot=slot)
    log.info("intent=%s slot=%s conf=%.3f lat=%.1fms -> %s",
             res.intent, slot, res.confidence, res.latency_ms, outcome.response)
    return {
        "acted": outcome.acted,
        "intent": res.intent,
        "category": outcome.category,
        "confidence": res.confidence,
        "latency_ms": res.latency_ms,
        "response": outcome.response,
        "phrase": phrase or (slot or ""),
        "slot": slot,
    }


def cmd_file(args):
    import soundfile as sf
    audio, sr = sf.read(args.file, dtype="float32")
    if sr != SAMPLE_RATE:
        import scipy.signal
        audio = scipy.signal.resample_poly(audio, SAMPLE_RATE, sr).astype(np.float32)
    clf = VCMClassifier(args.model, confidence_threshold=args.threshold)
    out = handle_command(SkillDispatcher(SimulatedSmartHome()), clf, audio,
                         phrase=args.phrase or "",
                         gate=not args.no_gate)
    print(json.dumps(out, indent=2))


def cmd_demo(args):
    """End-to-end demo over the held-out test set.

    Classifies every test clip (the real deployment path), then for each of
    the 19 in-scope intents reports how often it is predicted correctly and
    dispatches the model's most-confident clip for that intent to the
    simulated device. Nothing is cherry-picked: per-intent accuracy is the
    true test-set accuracy, and the full per-clip results are saved.
    """
    clf = VCMClassifier(args.model, confidence_threshold=args.threshold)
    device = SimulatedSmartHome()
    disp = SkillDispatcher(device)

    z = np.load(ROOT / "data" / "packs" / "test.npz", allow_pickle=True)
    ws, labels = z["waveforms"], z["labels"]
    intents = clf.intents

    # Real phrase text for each intent (from the dataset manifest), so the
    # slot parsers run on genuine command wording, not placeholders.
    import pandas as _pd
    man = _pd.read_csv(ROOT / "data" / "me2-dataset" / "manifest.csv")
    phrase_by_label = {
        lab: man.loc[man.command == lab, "transcript"].iloc[0]
        for lab in intents if (man.command == lab).any()
    }

    t0 = time.perf_counter()
    preds, confs, lats = [], [], []
    for w in ws:
        r = clf.classify(w)
        preds.append(r.intent_idx)
        confs.append(r.confidence)
        lats.append(r.latency_ms)
    classify_wall = time.perf_counter() - t0
    preds = np.array(preds)
    confs = np.array(confs)
    lats = np.array(lats)

    rows = []
    print(f"{'intent':<17}{'test acc':<12}{'best-conf pred':<20}{'conf':<8}{'ms':<8}response")
    for li in range(19):
        mask = labels == li
        if not mask.any():
            continue
        idxs = np.where(mask)[0]
        best = idxs[np.argmax(confs[idxs])]
        r = clf.classify(ws[best])
        real_phrase = phrase_by_label.get(intents[li], "")
        outcome = disp.dispatch(r.intent, phrase=real_phrase)
        n_correct = int(((preds[idxs]) == li).sum())
        rows.append({
            "true": intents[li], "n_clips": int(mask.sum()),
            "n_correct": n_correct, "acc": n_correct / int(mask.sum()),
            "best_clip_pred": r.intent, "best_clip_conf": float(r.confidence),
            "best_clip_lat_ms": float(r.latency_ms),
            "phrase": real_phrase,
            "response": outcome.response, "acted": outcome.acted,
        })
        print(f"{intents[li]:<17}{n_correct}/{mask.sum():<11}"
              f"{r.intent:<20}{r.confidence:<8.3f}{r.latency_ms:<8.1f}{outcome.response[:40]}")

    cats: dict[str, list] = {}
    for row in rows:
        cats.setdefault(CATEGORY_BY_INTENT[row["true"]], []).append(row)
    print("\nPer-category test accuracy:")
    cat_summary = {}
    for c, rs in sorted(cats.items()):
        tot = sum(r["n_clips"] for r in rs)
        cor = sum(r["n_correct"] for r in rs)
        cat_summary[c] = {"clips": tot, "correct": cor, "acc": round(cor / tot, 4)}
        print(f"  {c:<20}{cor}/{tot}  ({cor/tot:.0%})")

    inscope = labels != 19
    overall_correct = int((preds[inscope] == labels[inscope]).sum())
    overall_n = int(inscope.sum())
    print(f"\nOverall in-scope accuracy: {overall_correct}/{overall_n} "
          f"({overall_correct/overall_n:.1%})")
    print(f"Wall time: {classify_wall:.1f}s for {len(ws)} clips "
          f"({classify_wall/len(ws)*1000:.2f} ms/clip incl. features)")
    print(f"Model-only latency: p50={np.median(lats):.2f}ms "
          f"p95={np.percentile(lats,95):.2f}ms")

    print("\nFinal device state:")
    print(json.dumps(device.snapshot(), indent=2))

    out = ROOT / "results" / "stage6_demo.json"
    out.write_text(json.dumps({
        "rows": rows, "categories": cat_summary,
        "overall_inscope_acc": overall_correct / overall_n,
        "overall_correct": overall_correct, "overall_n": overall_n,
        "latency_ms": {"p50": float(np.median(lats)),
                       "p95": float(np.percentile(lats, 95)),
                       "wall_per_clip_ms": classify_wall / len(ws) * 1000},
        "device_state": device.snapshot(),
        "events": device.events,
    }, indent=2))
    print(f"\nsaved -> {out}")


# --- Command recording after the wake word ---------------------------------
SPEECH_RMS = 0.010        # a frame counts as voiced above this rms
MIC_FRAME = int(SAMPLE_RATE * 0.08)  # 80 ms mic frames (matches the input stream)


def record_command(stream, max_s: float = 2.0,
                   silence_s: float = 0.45) -> np.ndarray:
    """Record the command right after the wake word trigger.

    Energy-based endpointing: stop early after `silence_s` of trailing
    silence, hard-cap at `max_s`. The model crops to its 1 s window
    internally (loudest 1 s), so a short clip is fine.
    """
    chunks: list[np.ndarray] = []
    t0 = time.monotonic()
    voiced_until = None
    while time.monotonic() - t0 < max_s:
        f = np.asarray(stream.read(MIC_FRAME)[0]).flatten()
        chunks.append(f)
        if _rms(f) > SPEECH_RMS:
            voiced_until = time.monotonic()
        elif voiced_until is not None and time.monotonic() - voiced_until > silence_s:
            break
    audio = np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE, np.float32)
    return audio[: int(2.0 * SAMPLE_RATE)]


def cmd_mic(args):
    import sounddevice as sd
    from runtime.wakeword import WakeWordDetector

    clf = VCMClassifier(args.model, confidence_threshold=args.threshold)
    device = SimulatedSmartHome()
    disp = SkillDispatcher(device)
    ww = WakeWordDetector(wakeword=args.wakeword, threshold=args.ww_threshold)

    armed = {"flag": True}  # re-arm after each command

    print(f'ME2 VCM live mode: say "{args.wakeword.replace("_", " ")}".')
    print("Speak the command right after the wake word (up to 2 s).")
    print("Slots (time, color, brightness, etc.) are extracted by the model.")
    print('Ctrl-C to quit.')
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=MIC_FRAME) as stream:
        idle = 0
        _was_playing = False
        while True:
            frame = np.asarray(stream.read(MIC_FRAME)[0]).flatten()
            # Duck volume while music plays so the mic can hear the wake word.
            is_playing = device.state["music"]["playing"]
            if is_playing and not _was_playing:
                device.duck_volume()
            elif not is_playing and _was_playing:
                device.restore_volume()
            _was_playing = is_playing
            if armed["flag"]:
                ev = ww.detect(frame)
                if ev.triggered:
                    armed["flag"] = False
                    ww.reset()
                    print(f"\n*** wake word (p={ev.probability:.2f}) — listening ***")
                    # Restore volume so the user hears the response clearly.
                    device.restore_volume()
                    _was_playing = False
                    audio = record_command(stream)
                    out = handle_command(disp, clf, audio, gate=True)
                    if out.get("slot"):
                        print(f'    slot: {out["slot"]}')
                    if out.get("acted"):
                        print(">>> " + out["response"])
                    else:
                        print(">>> I didn't catch that.")
                    idle = 0
                    armed["flag"] = True
                    print("-" * 60)
            else:
                idle += 1
                if idle % 25 == 0:  # ~2 s housekeeping
                    for msg in device.tick():
                        print(">>> " + msg)


def main(argv=None):
    p = argparse.ArgumentParser(description="ME2 VCM runtime")
    p.add_argument("--model", type=str, default=str(DEFAULT_ONNX))
    p.add_argument("--threshold", type=float, default=0.35,
                   help="confidence gate (live-mic tuned to 0.35; Stage 5: 0.7 ~ 97%% in-scope precision)")
    p.add_argument("--no-gate", action="store_true")
    p.add_argument("--ww-threshold", type=float, default=0.5)
    p.add_argument("--wakeword", type=str, default="hey_mycroft",
                   help="openWakeWord pretrained model name (default: hey_mycroft)")
    p.add_argument("--phrase", type=str, default=None,
                   help="spoken-phrase override for slot parsing (--file mode)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mic", action="store_true")
    g.add_argument("--file", type=str)
    g.add_argument("--demo", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.file:
        cmd_file(args)
    elif args.demo:
        cmd_demo(args)
    elif args.mic:
        cmd_mic(args)


if __name__ == "__main__":
    main()
