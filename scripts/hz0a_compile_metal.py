"""Compile the native HZ-0A Metal kernel and emit an artifact hash."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def compile_kernel(source: Path, output: Path, sdk: str = "macosx") -> dict:
    metal = subprocess.run(["xcrun", "-sdk", sdk, "-f", "metal"], check=True, capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as temporary:
        air = Path(temporary) / "kernel.air"
        subprocess.run([metal, "-c", str(source), "-o", str(air)], check=True, capture_output=True, text=True)
        metallib = subprocess.run(["xcrun", "-sdk", sdk, "-f", "metallib"], check=True, capture_output=True, text=True).stdout.strip()
        subprocess.run([metallib, str(air), "-o", str(output)], check=True, capture_output=True, text=True)
    return {"source": str(source), "output": str(output), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "metallib_sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "compiled": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile the HZ-0A fused Metal forward kernel.")
    parser.add_argument("--source", type=Path, default=Path("restart/hz0a_pmetal/metal/gdn2_forward.metal"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(compile_kernel(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
