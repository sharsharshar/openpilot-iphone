"""Print the UDID of the newest available iPhone simulator (iOS >= 18). Reads `simctl list devices available -j` on stdin.
Set IOS_MAJOR (for example 18 or 26) to restrict the choice to that iOS major version."""
import json
import os
import re
import sys

want_major = os.environ.get("IOS_MAJOR", "").strip()

devices = json.load(sys.stdin)["devices"]
best = None
for runtime, devs in devices.items():
    m = re.search(r"iOS-(\d+)(?:-(\d+))?", runtime)
    if not m:
        continue
    ver = (int(m.group(1)), int(m.group(2) or 0))
    if ver[0] < 18:
        continue
    if want_major and ver[0] != int(want_major):
        continue
    for d in devs:
        if d.get("isAvailable", True) and d["name"].startswith("iPhone"):
            key = (ver, "Pro" not in d["name"], d["name"])
            if best is None or key > best[0]:
                best = (key, d["udid"], d["name"], runtime)
if best is None:
    sys.stderr.write("no available iPhone simulator with iOS >= 18 found\n")
    sys.exit(1)
sys.stderr.write("picked %s (%s)\n" % (best[2], best[3]))
print(best[1])
