"""Floor-plan image -> plan.json (see schema/floorplan.schema.json).

Default (hybrid): OpenCV measures the walls and a local vision model on your GPU
reads and locates the labels. Claude is called only if the result still fails
validation, and only when credentials are available.

    # Hybrid, local first (needs Ollama with a vision model)
    python extract/extract_plan.py brochure.jpg -o out/plan.json --crop 295,650,670,1085

    # Whole plan from one model
    python extract/extract_plan.py brochure.jpg -o out/plan.json --mode vlm --provider claude
    python extract/extract_plan.py brochure.jpg -o out/plan.json --mode vlm --provider ollama
"""
import argparse
import base64
import io
import json
import os
import pathlib
import sys
import urllib.request

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from validate import validate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = json.loads((ROOT / "schema" / "floorplan.schema.json").read_text(encoding="utf-8"))
PROMPT = (ROOT / "extract" / "prompt.md").read_text(encoding="utf-8")
SCHEMA.pop("$schema", None)
LOCAL_MODEL = "qwen3-vl:8b-instruct"


def prepare_image(path, crop, max_side):
    """Crop to one plan and cap the long side. Image tokens ~= width*height/750,
    so cropping a multi-plan brochure page is the single biggest saving."""
    im = Image.open(path).convert("RGB")
    if crop:
        im = im.crop(crop)
    # Small brochure crops are upscaled so printed labels stay legible
    scale = max_side / max(im.size)
    if scale < 1 or max(im.size) < max_side / 2:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode(), im.size


def claude_available():
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (pathlib.Path.home() / ".config" / "anthropic").exists())


def ask_claude(image_b64, model, effort, history=None, draft=None):
    import anthropic

    client = anthropic.Anthropic()
    ask = "Extract this floor plan."
    if draft:
        plan, errors = draft
        ask = ("A local pipeline measured the walls and produced this draft. Its geometry is usually right; "
               "fix what the validator reports and anything else that contradicts the image, then return the "
               "full plan.\n\nValidator errors:\n- " + "\n- ".join(errors) + "\n\nDraft:\n" + json.dumps(plan))
    messages = history or [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                # Cached so a correction retry does not pay for the image again
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": ask},
        ],
    }]
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=[{"type": "text", "text": PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        # If the model declines, the API retries on a fallback model in the same call
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {resp.stop_details}")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("output truncated; raise max_tokens")
    u = resp.usage
    print(
        f"[claude] {resp.model}: in={u.input_tokens} cache_read={u.cache_read_input_tokens} "
        f"cache_write={u.cache_creation_input_tokens} out={u.output_tokens}",
        file=sys.stderr,
    )
    text = next(b.text for b in resp.content if b.type == "text")
    messages = messages + [{"role": "assistant", "content": resp.content}]
    return json.loads(text), messages


def ask_ollama(image_b64, model, host):
    body = {
        "model": model,
        "stream": False,
        "format": SCHEMA,
        # Use an instruct (non-thinking) model: thinking models can reason for 20k+ tokens here
        "think": False,
        # Ollama's default context (4k on 16 GB cards) is too small for prompt + image + plan;
        # num_predict stops a looping model instead of hanging
        "options": {"temperature": 0, "num_ctx": 16384, "num_predict": 8192},
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Extract this floor plan.", "images": [image_b64]},
        ],
    }
    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        resp = json.loads(r.read())
    ev, secs = resp.get("eval_count", 0), resp.get("eval_duration", 1) / 1e9
    print(f"[ollama] {model}: prompt={resp.get('prompt_eval_count')} out={ev} tok at {ev / secs:.0f} tok/s", file=sys.stderr)
    if resp.get("done_reason") == "length":
        raise RuntimeError("output hit num_predict before the JSON closed; the model is probably looping")
    return json.loads(resp["message"]["content"])


def report(plan):
    errors, warnings, rep = validate(plan)
    print(f"[validate] {json.dumps(rep)}", file=sys.stderr)
    for w in warnings:
        print(f"  warn: {w}", file=sys.stderr)
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="plan.json")
    ap.add_argument("--crop", help="x0,y0,x1,y1 in source pixels, to isolate one plan on a brochure page")
    ap.add_argument("--max-side", type=int, default=1100, help="long side in px sent to the model")
    ap.add_argument("--mode", choices=["hybrid", "vlm"], default="hybrid",
                    help="hybrid: OpenCV geometry + local labels; vlm: one model reads the whole plan")
    ap.add_argument("--provider", choices=["claude", "ollama"], default="claude", help="model for --mode vlm")
    ap.add_argument("--model", help="Claude model id (default claude-opus-5-5)")
    ap.add_argument("--local-model", default=LOCAL_MODEL, help="Ollama vision model")
    ap.add_argument("--fallback", choices=["claude", "none"], default="claude",
                    help="hybrid only: ask Claude to correct a plan that fails validation")
    ap.add_argument("--no-furniture", action="store_true", help="hybrid only: skip furniture detection")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--retries", type=int, default=1, help="Claude correction passes when validation fails")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    args = ap.parse_args()

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    image_b64, size = prepare_image(args.image, crop, args.max_side)
    print(f"[image] {size[0]}x{size[1]} px, ~{size[0] * size[1] // 750} image tokens if sent to Claude", file=sys.stderr)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    claude_model = args.model or "claude-opus-5-5"

    history, plan, errors = None, None, []
    if args.mode == "hybrid":
        import cv2
        import numpy as np
        from hybrid import extract

        bgr = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        plan, debug = extract(bgr, args.local_model, args.ollama_host, furniture=not args.no_furniture)
        plan["meta"]["title"] = plan["meta"]["title"] or pathlib.Path(args.image).stem
        dbg_path = out.with_suffix(".debug.png")
        cv2.imwrite(str(dbg_path), debug)
        print(f"[hybrid] walls, cells and rooms drawn in {dbg_path}", file=sys.stderr)
        errors = report(plan)
        if errors and args.fallback == "claude":
            if claude_available():
                print(f"[fallback] {len(errors)} errors, sending the draft to Claude", file=sys.stderr)
                plan, history = ask_claude(image_b64, claude_model, args.effort, draft=(plan, errors))
                errors = report(plan)
            else:
                print("[fallback] skipped: no Claude credentials (set ANTHROPIC_API_KEY)", file=sys.stderr)

    attempts = 0 if args.mode == "hybrid" else args.retries + 1
    for attempt in range(attempts):
        if args.provider == "claude":
            plan, history = ask_claude(image_b64, claude_model, args.effort, history)
        else:
            plan = ask_ollama(image_b64, args.model or LOCAL_MODEL, args.ollama_host)
        errors = report(plan)
        if not errors or attempt == args.retries or args.provider != "claude":
            break
        # Text-only correction: the image is already in the (cached) history
        print(f"[retry] {len(errors)} errors, asking for a correction", file=sys.stderr)
        history.append({
            "role": "user",
            "content": "The plan failed validation:\n- " + "\n- ".join(errors) + "\nReturn the full corrected plan.",
        })

    for e in errors:
        print(f"  error: {e}", file=sys.stderr)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
