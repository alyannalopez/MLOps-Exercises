# Architecture E — PocketSphinx-style grammar-based command spotter

This is the **rule/grammar** option you asked about (how PocketSphinx works under
the hood). It's a fundamentally different paradigm from A–D (neural classifiers)
and worth keeping as a fallback because its model is genuinely <1 MB.

## How PocketSphinx does it
PocketSphinx (the embedded branch of CMU Sphinx) is a **GMM-HMM keyword-spotter**:

1. **Acoustic model** — a set of **Gaussian Mixture Models (GMMs)** over feature
   vectors (typically 13 MFCC + delta + delta-delta = 39-dim, or PLP). Each
   **phone** (or word) is a small **Hidden Markov Model** of 3–5 states; each
   state is a GMM of ~8–16 Gaussians.
2. **Language model** — a tiny **grammar / bigram** that enumerates the allowed
   words. For a fixed command set this is just a list of 10–20 phrases.
3. **Decoder** — a **Viterbi / beam search** over the product of acoustic +
   language models, frame by frame, emitting the best-scoring command (or silence).

Why it's small: a GMM-HMM for ~20 short words + a grammar is typically
**0.1–1 MB** of weights. That's why it runs on a Pi (and on microcontrollers)
with no GPU.

## To build our version
```
pip install pocketsphinx      # or build from source for the Pi
```
1. **Transcribe** the 10 command phrases (we already have the text in the
   manifest `text` column for optionB/snips/slurp).
2. **Train a GMM-HMM acoustic model** on the command words:
   ```
   sph2管 train ...            # (sphinx train) — or use the pretrained
                               # en-us GMM-HMM and adapt it
   ```
   Easiest path: use the **pretrained `en-us` GMM-HMM acoustic model** shipped
   with PocketSphinx and **adapt** it on our 10 phrases (MLLR / MAP adaptation).
3. **Write the grammar** (JSGF or plain list):
   ```
   #JSGF V1.0;
   grammar cmds;
   public <cmd> =
      ( play music | what is the weather | what time is it
      | turn on the light | turn off the light | dim the light
      | set a timer | set an alarm | set the temperature
      | pause | stop | next song | volume up | volume down
      | remind me | call mom );
   ```
4. **Run** the decoder on a 16 kHz mic stream; it emits the matched command.

## Trade-offs vs the neural models (A–D)
| | Grammar (E) | Neural (A–D) |
|---|---|---|
| Model size | **0.1–1 MB** | 0.4 KB – 2 MB |
| Latency (Pi) | ~10–50 ms | ~5–100 ms |
| Handles paraphrase | Poor (exact phrases only) | Good (learns variation) |
| Robust to noise | Moderate | Better (esp. with noisy aug) |
| New command | Edit grammar, re-adapt | Retrain |
| Dependency | `pocketsphinx` (C lib) | onnxruntime |

**Recommendation:** build A–D first (they generalize to paraphrase and are what
the exercise's "tiny neural model" framing expects). Keep E as the **lightest
possible fallback** and as a strong **baseline to beat** — if the neural model
can't clearly beat a grammar spotter on the benchmark, the grammar is the
simpler, more defensible choice.

## Files
- Grammar: `pocketsphinx/cmds.jsgf`
- Adapter script: `pocketsphinx/train_adapt.py` (runs after A–D are done)
