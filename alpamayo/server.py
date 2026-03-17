"""
Launch the Alpamayo-R1-10B vLLM inference server.

Usage:
    python -m alpamayo.server
    python -m alpamayo.server --model-path /raid/models/Alpamayo-R1-10B --port 8000
    python -m alpamayo.server --tensor-parallel 4  # use 4 GPUs
"""

import argparse
import subprocess
import sys


DEFAULT_MODEL = "nvidia/Alpamayo-R1-10B"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 16384


def build_vllm_command(
    model: str,
    host: str,
    port: int,
    tensor_parallel: int,
    max_model_len: int,
    dtype: str,
) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--served-model-name", "alpamayo",
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-model-len", str(max_model_len),
        "--dtype", dtype,
        "--trust-remote-code",
        "--enable-chunked-prefill",
    ]


def main():
    parser = argparse.ArgumentParser(description="Start Alpamayo-R1-10B vLLM server")
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL,
        help="Local path or HuggingFace model ID (default: nvidia/Alpamayo-R1-10B)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum sequence length (default: 16384)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32", "auto"],
        help="Model weight dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    cmd = build_vllm_command(
        model=args.model_path,
        host=args.host,
        port=args.port,
        tensor_parallel=args.tensor_parallel,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
    )

    print(f"[server] Starting Alpamayo-R1-10B on {args.host}:{args.port}")
    print(f"[server] Model: {args.model_path}")
    print(f"[server] GPUs: {args.tensor_parallel}  dtype: {args.dtype}")
    print(f"[server] Command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[server] Stopped.")
    except subprocess.CalledProcessError as e:
        print(f"[server] Server exited with code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
