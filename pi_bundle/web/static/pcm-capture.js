// AudioWorklet node: streams raw Float32 mono audio to the main thread.
class PCMCapture extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0), [ch.buffer]);
    return true;
  }
}
registerProcessor('pcm-capture', PCMCapture);
