"""Generate a synthetic 'hey mycroft' true-positive clip with macOS `say`."""
import subprocess
import sys

subprocess.run(
    ["say", "-v", "Samantha", "Hey Microft, please turn on the lights",
     "-o", "/tmp/hey_mycroft.aiff"],
    check=True)
subprocess.run(
    ["afconvert", "-d", "LEI16@16000", "-f", "WAVE",
     "/tmp/hey_mycroft.aiff", "/tmp/hey_mycroft.wav"],
    check=True)
print("wrote /tmp/hey_mycroft.wav")
