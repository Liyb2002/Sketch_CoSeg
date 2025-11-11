import os
import json
import base64
from typing import List, Tuple
from openai import OpenAI

# ---------- CONFIG ----------
INPUT_DIR = "inputs"

# Lighter, vision-capable models for this task
MODEL_STEP1 = "gpt-4.1"       # for object_type + components detection
MODEL_STEP2 = "gpt-4.1-mini"  # for per-image counting

client = OpenAI()

# ---------- HELPERS ----------

def get_image_files(input_dir: str) -> List[str]:
    """
    Collect all image files from INPUT_DIR.
    Ignores non-image files and components*.json outputs.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = []

    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if not os.path.isfile(path):
            continue
        root, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            continue
        if root.startswith("components"):
            continue
        files.append(name)

    if not files:
        raise FileNotFoundError(f"No image files found in {input_dir}")

    # Stable ordering: numeric if possible, else lexicographic
    def sort_key(fname: str):
        stem = os.path.splitext(fname)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    files.sort(key=sort_key)
    return files


def guess_mime_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def encode_image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    mime = guess_mime_type(path)
    return f"data:{mime};base64,{b64}"


def gpt_json_from_response(response) -> dict:
    """
    Extracts the first text block from Responses API output and parses as JSON.
    """
    if not hasattr(response, "output") or not response.output:
        raise RuntimeError(f"No output field in response: {response}")

    text = None

    for item in response.output:
        content_list = getattr(item, "content", None) or []
        for block in content_list:
            if getattr(block, "type", None) == "output_text":
                txt_obj = getattr(block, "text", None)
                if txt_obj is None:
                    continue
                value = getattr(txt_obj, "value", None)
                if value is None:
                    value = txt_obj if isinstance(txt_obj, str) else str(txt_obj)
                text = value
                break
        if text:
            break

    if not text:
        raise RuntimeError(f"Failed to locate text output in model response. Raw: {response}")

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse JSON from model response: {e}\nRaw text was:\n{text}"
        )

# ---------- STEP 1 ----------
# Detect common object type + up to 10 components

def detect_object_and_components(image_files: List[str]) -> Tuple[str, List[str]]:
    content_blocks = [
        {
            "type": "input_text",
            "text": (
                "You are analyzing multiple sketches. "
                "All images depict the SAME TYPE of object (e.g., all cars, all planes, all motorbikes), "
                "but with different styles or configurations.\n\n"
                "TASK:\n"
                "1. Decide the single common object_type that best describes ALL images.\n"
                "2. Infer a concise list of visually identifiable components that belong to that object_type "
                "and could reasonably appear in these sketches.\n"
                "3. Return ONLY valid JSON in this exact format:\n"
                "{\n"
                "  \"object_type\": \"...\",\n"
                "  \"components\": [\"component_name_1\", \"component_name_2\", ...]\n"
                "}\n"
                "Requirements:\n"
                "- components: short, lowercase, snake_case or simple words (e.g. \"wheel\", \"wing\", \"door_handle\").\n"
                "- Only include components that can be visually distinguished in a sketch.\n"
                "- You MUST NOT list more than 10 components in total.\n"
                "- No extra explanation, no markdown, JSON only."
            ),
        }
    ]

    for filename in image_files:
        path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        content_blocks.append(
            {
                "type": "input_image",
                "image_url": encode_image_to_data_url(path),
            }
        )

    response = client.responses.create(
        model=MODEL_STEP1,
        input=[
            {"role": "user", "content": content_blocks}
        ],
    )

    result = gpt_json_from_response(response)
    object_type = result.get("object_type")
    components = result.get("components", [])

    if not object_type or not isinstance(components, list):
        raise ValueError(f"Invalid JSON from step 1: {result}")

    components = [str(c).strip() for c in components if str(c).strip()][:10]
    return object_type, components

# ---------- STEP 2 ----------
# For each image: count components and output JSON with object_name

def analyze_image_components(object_type: str, components: List[str], image_files: List[str]) -> None:
    for filename in image_files:
        index = os.path.splitext(filename)[0]
        path = os.path.join(INPUT_DIR, filename)

        content_blocks = [
            {
                "type": "input_text",
                "text": (
                    f"You are given one sketch of a {object_type}.\n"
                    f"Here is the fixed list of allowed components (at most 10):\n"
                    f"{components}\n\n"
                    "TASK:\n"
                    "For THIS image only, count how many of each listed component are present.\n"
                    "Rules:\n"
                    "- You MUST NOT introduce new component names.\n"
                    "- If a component from the list is not present, set its count to 0.\n"
                    "- Only count components clearly visible in the sketch.\n"
                    "- Return ONLY valid JSON in this exact format:\n"
                    "{\n"
                    f"  \"image\": \"{filename}\",\n"
                    f"  \"object_name\": \"{object_type}\",\n"
                    "  \"components\": [\n"
                    "    { \"name\": \"component_name_1\", \"count\": number },\n"
                    "    { \"name\": \"component_name_2\", \"count\": number }\n"
                    "  ]\n"
                    "}\n"
                    "- No extra keys, no markdown, no explanation."
                ),
            },
            {
                "type": "input_image",
                "image_url": encode_image_to_data_url(path),
            },
        ]

        response = client.responses.create(
            model=MODEL_STEP2,
            input=[
                {"role": "user", "content": content_blocks}
            ],
        )

        result = gpt_json_from_response(response)

        if result.get("image") != filename:
            raise ValueError(f"Invalid JSON for {filename}: missing or wrong image key.")
        raw_components = result.get("components")

        # Normalize the components list
        name_to_count = {}
        if isinstance(raw_components, dict):
            for name, val in raw_components.items():
                name_to_count[str(name).strip().lower()] = val
        elif isinstance(raw_components, list):
            for entry in raw_components:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "")).strip().lower()
                if not name:
                    continue
                name_to_count[name] = entry.get("count", 0)
        else:
            raise ValueError(f"Invalid components structure for {filename}: {type(raw_components)}")

        clean_list = []
        for comp in components:
            key = comp.lower()
            raw_val = name_to_count.get(key, 0)
            try:
                val = int(raw_val)
            except Exception:
                val = 0
            if val < 0:
                val = 0
            clean_list.append({"name": comp, "count": val})

        output = {
            "image": filename,
            "object_name": object_type,
            "components": clean_list,
        }

        out_path = os.path.join(INPUT_DIR, f"components_{index}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Wrote {out_path}")

# ---------- MAIN FLOW ----------

if __name__ == "__main__":
    image_files = get_image_files(INPUT_DIR)
    print(f"Processing {len(image_files)} image(s): {image_files}")
    obj_type, components = detect_object_and_components(image_files)
    print("Detected object_type:", obj_type)
    print("Detected components (max 10):", components)
    analyze_image_components(obj_type, components, image_files)
