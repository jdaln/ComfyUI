#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

try:
    from PIL import Image
except ImportError:  # pragma: no cover - Pillow is expected in the runtime image.
    Image = None


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ARTIFACT_KEYS = ("images", "audio", "gifs", "files")


def deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(override)


def classify_blueprint(name: str) -> dict[str, str]:
    family = "other"
    input_kind = "none"
    output_kind = "artifact"

    if name.startswith("Text to Image"):
        family = "text-to-image"
        input_kind = "text"
        output_kind = "image"
    elif name.startswith("Text to Video"):
        family = "text-to-video"
        input_kind = "text"
        output_kind = "video"
    elif name.startswith("Image to Video"):
        family = "image-to-video"
        input_kind = "image"
        output_kind = "video"
    elif name.startswith("First-Last-Frame to Video"):
        family = "video-interpolation"
        input_kind = "image-pair"
        output_kind = "video"
    elif name.startswith("Text to Audio"):
        family = "text-to-audio"
        input_kind = "text"
        output_kind = "audio"
    elif name.startswith("Video Captioning"):
        family = "video-analysis"
        input_kind = "video"
        output_kind = "text"
    elif name.startswith("Video "):
        family = "video-utility"
        input_kind = "video"
        output_kind = "video"
    elif name.startswith("Prompt Enhance"):
        family = "text-utility"
        input_kind = "text"
        output_kind = "text"
    elif name.startswith("Image Captioning"):
        family = "image-analysis"
        input_kind = "image"
        output_kind = "text"
    elif name.startswith("Image to Model"):
        family = "image-analysis"
        input_kind = "image"
        output_kind = "model"
    elif name.startswith("Image to Depth Map"):
        family = "image-analysis"
        input_kind = "image"
        output_kind = "image"
    elif name.startswith("Image to Layers"):
        family = "image-analysis"
        input_kind = "image"
        output_kind = "artifact"
    elif name.startswith("Image Outpainting"):
        family = "image-generation"
        input_kind = "image"
        output_kind = "image"
    elif name.startswith("Image Inpainting") or name.startswith("Image Edit") or name.startswith("Image Upscale"):
        family = "image-generation"
        input_kind = "image"
        output_kind = "image"
    elif name.startswith("Pose to Image") or name.startswith("Depth to Image") or name.startswith("Canny to Image"):
        family = "image-generation"
        input_kind = "image"
        output_kind = "image"
    elif name.startswith("Pose to Video") or name.startswith("Depth to Video") or name.startswith("Canny to Video"):
        family = "video-generation"
        input_kind = "image"
        output_kind = "video"
    elif name in {
        "Brightness and Contrast",
        "Chromatic Aberration",
        "Color Adjustment",
        "Color Balance",
        "Color Curves",
        "Crop Images 2x2",
        "Crop Images 3x3",
        "Edge-Preserving Blur",
        "Film Grain",
        "Glow",
        "Hue and Saturation",
        "Image Blur",
        "Image Channels",
        "Image Levels",
        "Sharpen",
        "Unsharp Mask",
    }:
        family = "image-utility"
        input_kind = "image"
        output_kind = "image"

    return {
        "family": family,
        "input_kind": input_kind,
        "output_kind": output_kind,
    }


def load_manifest(manifest_path: Path) -> dict:
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def materialize_entries(repo_root: Path, manifest: dict) -> list[dict]:
    blueprint_root = repo_root / manifest["blueprint_root"]
    defaults = manifest.get("defaults", {})
    overrides = manifest.get("entries", {})
    entries = []
    seen = set()

    for path in sorted(blueprint_root.glob("*.json")):
        name = path.stem
        base_entry = {
            "name": name,
            "blueprint": f"{manifest['blueprint_root']}/{path.name}",
            **classify_blueprint(name),
            "status": defaults.get("status", "todo"),
            "block_reason": None,
            "smoke": copy.deepcopy(defaults.get("smoke", {})),
            "prerequisites": copy.deepcopy(defaults.get("prerequisites", {})),
            "notes": [],
        }
        entry = deep_merge(base_entry, overrides.get(path.name, {}))
        if entry["smoke"].get("expected_output") is None:
            entry["smoke"]["expected_output"] = entry["output_kind"]
        entries.append(entry)
        seen.add(path.name)

    unknown = sorted(set(overrides) - seen)
    if unknown:
        raise SystemExit(f"Manifest overrides refer to missing blueprints: {', '.join(unknown)}")

    return entries


def parse_templates(raw_values: list[str]) -> set[str]:
    selected = set()
    for raw_value in raw_values:
        for piece in raw_value.split(","):
            piece = piece.strip()
            if piece:
                selected.add(piece)
    return selected


def filter_entries(entries: list[dict], selected_templates: set[str]) -> list[dict]:
    if not selected_templates:
        return entries

    filtered = []
    for entry in entries:
        blueprint_name = Path(entry["blueprint"]).name
        if entry["name"] in selected_templates or blueprint_name in selected_templates:
            filtered.append(entry)
    return filtered


def queue_prompt(base_url: str, prompt: dict) -> str:
    request = urllib.request.Request(
        f"{base_url}/prompt",
        data=json.dumps({"prompt": prompt}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["prompt_id"]


def get_history(base_url: str, prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/history/{prompt_id}", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_completion(base_url: str, prompt_id: str, timeout_seconds: int) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        history = get_history(base_url, prompt_id)
        record = history.get(prompt_id)
        if record:
            status = record.get("status", {})
            if status.get("completed"):
                if status.get("status_str") != "success":
                    raise RuntimeError(f"Prompt {prompt_id} completed with status {status.get('status_str')}")
                return record
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id} after {timeout_seconds} seconds")


def refresh_completed_record(base_url: str, prompt_id: str, record: dict, retries: int = 5, delay_seconds: float = 1.0) -> dict:
    refreshed = record
    for _ in range(retries):
        if refreshed.get("outputs"):
            return refreshed
        time.sleep(delay_seconds)
        history = get_history(base_url, prompt_id)
        refreshed = history.get(prompt_id, refreshed)
    return refreshed


def fetch_output_bytes(base_url: str, artifact: dict) -> bytes:
    query = urllib.parse.urlencode(
        {
            "filename": artifact["filename"],
            "subfolder": artifact.get("subfolder", ""),
            "type": artifact.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=30) as response:
        return response.read()


def validate_image_bytes(filename: str, payload: bytes) -> None:
    if Image is None:
        if not payload:
            raise RuntimeError(f"Image output is empty: {filename}")
        return

    with Image.open(BytesIO(payload)) as image:
        image.load()
        if image.getbbox() is None:
            raise RuntimeError(f"Image output appears blank: {filename}")


def artifact_kind(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "artifact"


def validate_outputs(base_url: str, entry: dict, record: dict) -> list[dict]:
    outputs = record.get("outputs", {})
    if not outputs:
        raise RuntimeError(f"No outputs recorded for {entry['name']}")

    summaries = []
    matched_expected_output = False
    expected_output = entry["smoke"]["expected_output"]

    for node_id, node_output in outputs.items():
        node_had_artifact = False
        for key in ARTIFACT_KEYS:
            for artifact in node_output.get(key, []):
                if not isinstance(artifact, dict) or "filename" not in artifact:
                    continue
                payload = fetch_output_bytes(base_url, artifact)
                if not payload:
                    raise RuntimeError(f"Downloaded empty artifact for {entry['name']}: {artifact['filename']}")

                kind = artifact_kind(artifact["filename"])
                if kind == "image":
                    validate_image_bytes(artifact["filename"], payload)
                if kind == expected_output or expected_output == "artifact":
                    matched_expected_output = True

                summaries.append(
                    {
                        "node_id": node_id,
                        "artifact_key": key,
                        "filename": artifact["filename"],
                        "size": len(payload),
                        "kind": kind,
                    }
                )
                node_had_artifact = True

        if not node_had_artifact and node_output:
            summaries.append(
                {
                    "node_id": node_id,
                    "artifact_key": "metadata",
                    "keys": sorted(node_output.keys()),
                }
            )
            if expected_output == "text":
                matched_expected_output = True

    if not summaries:
        raise RuntimeError(f"No usable output artifacts recorded for {entry['name']}")
    if not matched_expected_output:
        raise RuntimeError(
            f"Outputs for {entry['name']} did not include the expected output kind: {expected_output}"
        )

    return summaries


def ensure_fixtures(repo_root: Path, entry: dict) -> None:
    missing = []
    input_root = repo_root / "input"
    for fixture in entry["smoke"].get("fixtures", []):
        if not (input_root / fixture).exists():
            missing.append(fixture)
    if missing:
        raise FileNotFoundError(
            f"Missing required fixture(s) for {entry['name']}: {', '.join(missing)}"
        )


def uniquify_filename_prefixes(prompt: dict, suffix: str) -> None:
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        prefix = inputs.get("filename_prefix")
        if isinstance(prefix, str) and prefix:
            inputs["filename_prefix"] = f"{prefix}_{suffix}"


def run_smoke(repo_root: Path, base_url: str, entry: dict, timeout_seconds: int) -> dict:
    ensure_fixtures(repo_root, entry)

    graph_path = repo_root / entry["smoke"]["api_graph"]
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing API graph for {entry['name']}: {graph_path}")

    with graph_path.open(encoding="utf-8") as handle:
        prompt = json.load(handle)

    uniquify_filename_prefixes(prompt, str(int(time.time() * 1000)))

    prompt_id = queue_prompt(base_url, prompt)
    record = wait_for_completion(base_url, prompt_id, timeout_seconds)
    record = refresh_completed_record(base_url, prompt_id, record)
    artifacts = validate_outputs(base_url, entry, record)
    return {
        "name": entry["name"],
        "prompt_id": prompt_id,
        "status": record.get("status", {}).get("status_str"),
        "artifacts": artifacts,
    }


def print_entry_table(entries: list[dict]) -> None:
    counts = {}
    enabled = 0
    for entry in entries:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        if entry["smoke"].get("enabled") and entry["smoke"].get("api_graph"):
            enabled += 1

    print(f"Bundled blueprints: {len(entries)}")
    print(f"Smoke-enabled entries: {enabled}")
    print("Status counts:")
    for status in sorted(counts):
        print(f"  {status}: {counts[status]}")
    print()
    print(f"{'STATUS':<10} {'SMOKE':<7} {'FAMILY':<20} NAME")
    for entry in entries:
        smoke_state = "enabled" if entry["smoke"].get("enabled") and entry["smoke"].get("api_graph") else "-"
        print(f"{entry['status']:<10} {smoke_state:<7} {entry['family']:<20} {entry['name']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List or run container-only smoke coverage for bundled ComfyUI blueprints."
    )
    parser.add_argument(
        "--manifest",
        default="tests/inference/bundled_template_coverage.json",
        help="Manifest path relative to the ComfyUI repo root.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8188",
        help="Base URL for the ComfyUI HTTP API.",
    )
    parser.add_argument(
        "--template",
        action="append",
        default=[],
        help="Template name or blueprint filename to include. Can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the materialized bundled-template coverage matrix instead of running smoke tests.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON when used with --list.",
    )
    parser.add_argument(
        "--run-all-smokes",
        action="store_true",
        help="Run every entry that has an API graph, even if its smoke entry is not enabled yet.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=900,
        help="Per-template timeout while waiting for ComfyUI history completion.",
    )
    parser.add_argument(
        "--report-path",
        help="Optional path for a JSON result report.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = repo_root / args.manifest
    manifest = load_manifest(manifest_path)
    entries = materialize_entries(repo_root, manifest)
    entries = filter_entries(entries, parse_templates(args.template))

    if not entries:
        parser.error("No bundled templates matched the provided filters.")

    if args.json and not args.list:
        parser.error("--json currently requires --list.")

    if args.list:
        if args.json:
            print(json.dumps(entries, indent=2))
            return 0
        print_entry_table(entries)
        return 0

    runnable = []
    for entry in entries:
        smoke = entry.get("smoke", {})
        if entry["status"] == "blocked":
            print(f"BLOCKED  {entry['name']}: {entry['block_reason']}", file=sys.stderr)
            continue
        if not smoke.get("api_graph"):
            print(f"SKIP     {entry['name']}: no API graph is tracked yet", file=sys.stderr)
            continue
        if not smoke.get("enabled") and not args.run_all_smokes:
            print(f"SKIP     {entry['name']}: smoke entry is not enabled yet", file=sys.stderr)
            continue
        runnable.append(entry)

    if not runnable:
        print("No runnable bundled-template smokes matched the current selection.", file=sys.stderr)
        return 1

    results = []
    failures = []
    for entry in runnable:
        print(f"RUN      {entry['name']}")
        try:
            result = run_smoke(repo_root, args.base_url, entry, args.timeout_seconds)
        except Exception as exc:  # pragma: no cover - exercised by live smoke execution.
            failures.append({"name": entry["name"], "error": str(exc)})
            print(f"FAIL     {entry['name']}: {exc}", file=sys.stderr)
            continue

        results.append(result)
        print(f"PASS     {entry['name']} ({result['prompt_id']})")

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"results": results, "failures": failures}, indent=2),
            encoding="utf-8",
        )

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())