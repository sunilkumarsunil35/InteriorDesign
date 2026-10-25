"""Floor-plan image -> plan.json (see schema/floorplan.schema.json).

The image is read once by a vision model, which fills the schema. Everything
after that (validation, 3D build, walkthrough) runs locally with no AI.

    # Claude (needs ANTHROPIC_API_KEY or `ant auth login`)
    python extract/extract_plan.py brochure.jpg -o out/plan.json --crop 295,650,670,1085

    # Fully local, via Ollama with a vision model
    python extract/extract_plan.py brochure.jpg -o out/plan.json --provider ollama --model qwen2.5vl:7b
"""
import argparse
import base64
import io
import json
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


def ask_claude(image_b64, model, effort, history=None):
    import anthropic

    client = anthropic.Anthropic()
    messages = history or [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": image_b64},
                # Cached so a correction retry does not pay for the image again
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "Extract this floor plan."},
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
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Extract this floor plan.", "images": [image_b64]},
        ],
    }
    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.loads(r.read())["message"]["content"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="plan.json")
    ap.add_argument("--crop", help="x0,y0,x1,y1 in source pixels, to isolate one plan on a brochure page")
    ap.add_argument("--max-side", type=int, default=1100, help="long side in px sent to the model")
    ap.add_argument("--provider", choices=["claude", "ollama"], default="claude")
    ap.add_argument("--model", help="default: claude-opus-5-5 / qwen2.5vl:7b")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--retries", type=int, default=1, help="correction passes when validation fails")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    args = ap.parse_args()

    crop = tuple(int(v) for v in args.crop.split(",")) if args.crop else None
    image_b64, size = prepare_image(args.image, crop, args.max_side)
    print(f"[image] {size[0]}x{size[1]} px, ~{size[0] * size[1] // 750} image tokens", file=sys.stderr)

    history = None
    for attempt in range(args.retries + 1):
        if args.provider == "claude":
            plan, history = ask_claude(image_b64, args.model or "claude-opus-5-5", args.effort, history)
        else:
            plan = ask_ollama(image_b64, args.model or "qwen2.5vl:7b", args.ollama_host)

        errors, warnings, report = validate(plan)
        print(f"[validate] {json.dumps(report)}", file=sys.stderr)
        for w in warnings:
            print(f"  warn: {w}", file=sys.stderr)
        if not errors or attempt == args.retries or args.provider != "claude":
            break
        # Text-only correction: the image is already in the (cached) history
        print(f"[retry] {len(errors)} errors, asking for a correction", file=sys.stderr)
        history.append({
            "role": "user",
            "content": "The plan failed validation:\n- " + "\n- ".join(errors)
            + "\nReturn the full corrected plan.",
        })

    for e in errors:
        print(f"  error: {e}", file=sys.stderr)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"[done] wrote {out}", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
