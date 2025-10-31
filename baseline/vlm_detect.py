# vlm_detect.py
# Exhaustive part discovery from a single sketch image.
# Input : sketch/0.png
# Output: sketch/out/0.components.json  ->  {"components": [ ... many fine-grained parts ... ]}

import os, io, json, base64, re
from pathlib import Path
import requests
from PIL import Image

# ---------------- Config ----------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_URL     = "https://api.openai.com/v1/chat/completions"
MODEL          = "gpt-4o-mini"  # vision-capable

IN_PATH  = Path("sketch/0.png")
OUT_DIR  = Path("sketch/out")
OUT_JSON = OUT_DIR / "0.components.json"

# Allow very detailed outputs
MAX_PARTS = 64

# Extremely small blocklist: only obvious non-part terms
BLOCKLIST = {
    "background","outline","silhouette","shadow","texture","pattern",
    "line","stroke","sketch","drawing","object"
}

def _to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\-_/ ()]", "", s)
    s = re.sub(r"\s+", " ", s)
    # normalize simple plurals to singles when it looks like a generic class
    for base in ("leg","arm","wheel","spoke","pedal","handle","bolt","screw","bar","tube","rod"):
        if s == base + "s": s = base
    return s

def main():
    if not OPENAI_API_KEY:
        raise SystemExit("ERROR: OPENAI_API_KEY not set")

    if not IN_PATH.exists():
        raise SystemExit(f"ERROR: Input not found: {IN_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img = Image.open(IN_PATH).convert("RGB")
    data_uri = _to_data_uri(img)

    # Prompt: exhaustive, fine-grained, no category guessing, side-specific allowed
    system_prompt = (
        "You extract semantic PART NAMES from a single line drawing. "
        "Do NOT guess or state the object category. "
        "Return ONLY parts that are visually present. "
        "Be EXHAUSTIVE: include large parts, subparts, connectors, braces, joints, hubs, axles, spokes, "
        "bearings, brackets, pedals, cranks, chains, frames, forks, seats, seat-posts, stems, handlebars, grips, "
        "tires, rims, spokes, brakes, levers, cables, guards, racks, fenders, pegs, hinges, screws, bolts, nuts, links, etc., "
        "but ONLY if they appear in the drawing. "
        "Prefer short, generic, human-readable names. "
        "If left/right (or front/rear) instances are clearly visible, list them separately "
        "(e.g., 'left wheel', 'right wheel'). "
        f"Return at most {MAX_PARTS} items. "
        "If uncertain about a part, omit it. "
        "Output must be a JSON array of strings, nothing else."
    )

    user_prompt = (
        "List every visible part and subpart you can identify in this sketch. "
        "No category names, no background terms. "
        "Return ONLY a JSON array of strings."
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]}
        ]
    }

    resp = requests.post(OPENAI_URL, headers=headers, data=json.dumps(payload), timeout=90)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Parse JSON array; be tolerant if model adds stray text
    parts = []
    try:
        # Try direct
        parts = json.loads(raw)
        if not isinstance(parts, list):
            parts = []
    except Exception:
        # Best-effort extract JSON array
        try:
            start = raw.find("[")
            end   = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                parts = json.loads(raw[start:end+1])
                if not isinstance(parts, list):
                    parts = []
        except Exception:
            parts = []

    # Normalize, filter, dedupe (preserve order)
    cleaned = []
    seen = set()
    for p in parts:
        p = _normalize_label(str(p))
        if not p or p in BLOCKLIST:
            continue
        if p not in seen:
            seen.add(p)
            cleaned.append(p)

    # Cap to MAX_PARTS
    cleaned = cleaned[:MAX_PARTS]

    OUT_JSON.write_text(json.dumps({"components": cleaned}, indent=2))
    print(f"Wrote components → {OUT_JSON}")
    print(cleaned)

if __name__ == "__main__":
    main()
