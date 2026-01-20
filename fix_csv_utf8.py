from pathlib import Path

FILES = ["scene_library.csv", "dialogue_library.csv", "disclaimer_prompt2.csv"]

def fix_text(s: str) -> str:
    if not s:
        return s
    # repair common mojibake where UTF-8 bytes were decoded as latin1
    if "Ã" in s or "Â" in s or "Ä" in s:
        try:
            s2 = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if s2 and s2.count("Ã") < s.count("Ã"):
                return s2
        except Exception:
            pass
    return s

for fn in FILES:
    p = Path(fn)
    if not p.exists():
        print("Missing:", fn)
        continue

    raw = p.read_text(encoding="utf-8", errors="ignore")

    # Split lines, fix each line, write back UTF-8
    lines = raw.splitlines()
    fixed_lines = [fix_text(line) for line in lines]
    fixed = "\n".join(fixed_lines) + ("\n" if raw.endswith("\n") else "")

    p.write_text(fixed, encoding="utf-8")
    print("Fixed:", fn)

print("DONE")
