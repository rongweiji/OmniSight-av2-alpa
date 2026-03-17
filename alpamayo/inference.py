"""
End-to-end AV2 scene inference pipeline.

Loads an AV2 scene with load_scene.py, formats it into a structured prompt,
calls Alpamayo-R1-10B, and returns the model's explanation.

Usage (Python):
    from alpamayo import SceneInference

    inf = SceneInference(server_url="http://localhost:8000/v1")
    result = inf.run(data_dir="/raid/av2/sensor/val", log_id="scene-001")
    print(result.explanation)

Usage (CLI):
    python -m alpamayo.inference \
        --data-dir /raid/av2/sensor/val \
        --log-id <scene_id> \
        --task summary
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .client import AlpamayoClient, GenerationConfig
from .prompts import (
    SYSTEM_PROMPT,
    custom_prompt,
    lidar_density_prompt,
    object_behavior_prompt,
    scene_summary_prompt,
)


TASK_CHOICES = ["summary", "behavior", "lidar", "custom"]


@dataclass
class InferenceResult:
    log_id: str
    city_name: str
    task: str
    prompt_used: str
    explanation: str

    def display(self):
        """Pretty-print the result to stdout."""
        sep = "=" * 70
        print(sep)
        print(f"Scene : {self.log_id}")
        print(f"City  : {self.city_name}")
        print(f"Task  : {self.task}")
        print(sep)
        print(self.explanation)
        print(sep)


class SceneInference:
    """
    Orchestrates loading an AV2 scene and querying Alpamayo-R1-10B.

    Args:
        server_url:  Base URL of the running vLLM server.
        model_name:  Served model name (must match server --served-model-name).
        config:      Generation hyperparameters.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model_name: str = "alpamayo",
        config: GenerationConfig | None = None,
    ):
        self.client = AlpamayoClient(
            base_url=server_url,
            model_name=model_name,
            config=config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data_dir: str,
        log_id: str | None = None,
        task: str = "summary",
        question: str | None = None,
        category: str | None = None,
        stream: bool = False,
    ) -> InferenceResult:
        """
        Full pipeline: load scene → build prompt → call model → return result.

        Args:
            data_dir:  Path to AV2 split directory (e.g. /raid/av2/sensor/val).
            log_id:    Scene log ID. If None, uses the first scene in data_dir.
            task:      One of 'summary', 'behavior', 'lidar', 'custom'.
            question:  Required when task='custom'. Free-form question string.
            category:  Optional object category filter for task='behavior'.
            stream:    If True, stream response tokens to stdout.

        Returns:
            InferenceResult with the model's explanation.
        """
        # 1. Load AV2 scene
        scene = self._load_scene(data_dir, log_id)

        # 2. Build prompt
        user_prompt = self._build_prompt(scene, task, question, category)

        # 3. Call model
        if stream:
            explanation = self._call_stream(user_prompt)
        else:
            explanation = self.client.explain(SYSTEM_PROMPT, user_prompt)

        return InferenceResult(
            log_id=scene["log_id"],
            city_name=scene.get("city_name", "unknown"),
            task=task,
            prompt_used=user_prompt,
            explanation=explanation,
        )

    def run_on_scene(
        self,
        scene: dict,
        task: str = "summary",
        question: str | None = None,
        category: str | None = None,
        stream: bool = False,
    ) -> InferenceResult:
        """
        Run inference on a pre-loaded scene dict (output of load_scene.py).

        Useful when you have already loaded the scene and want to run
        multiple tasks without reloading.
        """
        user_prompt = self._build_prompt(scene, task, question, category)

        if stream:
            explanation = self._call_stream(user_prompt)
        else:
            explanation = self.client.explain(SYSTEM_PROMPT, user_prompt)

        return InferenceResult(
            log_id=scene["log_id"],
            city_name=scene.get("city_name", "unknown"),
            task=task,
            prompt_used=user_prompt,
            explanation=explanation,
        )

    def check_server(self) -> bool:
        """Return True if the Alpamayo server is reachable."""
        return self.client.is_ready()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_scene(data_dir: str, log_id: str | None) -> dict:
        # Import here so alpamayo module works without av2 installed
        # (e.g. on a machine that only runs the client side)
        try:
            from load_scene import load_scene
        except ImportError:
            # Try relative import when running from project root
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from load_scene import load_scene  # type: ignore

        return load_scene(data_dir, log_id)

    @staticmethod
    def _build_prompt(
        scene: dict,
        task: str,
        question: str | None,
        category: str | None,
    ) -> str:
        if task == "summary":
            return scene_summary_prompt(scene)
        elif task == "behavior":
            return object_behavior_prompt(scene, category=category)
        elif task == "lidar":
            return lidar_density_prompt(scene)
        elif task == "custom":
            if not question:
                raise ValueError("task='custom' requires a --question argument.")
            return custom_prompt(scene, question)
        else:
            raise ValueError(f"Unknown task: {task!r}. Choose from {TASK_CHOICES}")

    def _call_stream(self, user_prompt: str) -> str:
        chunks = []
        for chunk in self.client.explain_stream(SYSTEM_PROMPT, user_prompt):
            print(chunk, end="", flush=True)
            chunks.append(chunk)
        print()  # newline after stream ends
        return "".join(chunks)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-R1-10B inference on an AV2 scene"
    )
    parser.add_argument("--data-dir", required=True, help="Path to AV2 split dir")
    parser.add_argument("--log-id", default=None, help="Scene log ID (optional)")
    parser.add_argument(
        "--task",
        default="summary",
        choices=TASK_CHOICES,
        help="Inference task (default: summary)",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Custom question (required when --task=custom)",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter objects by category (for --task=behavior)",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream response tokens to stdout",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Max tokens to generate"
    )
    args = parser.parse_args()

    config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    inf = SceneInference(server_url=args.server_url, config=config)

    if not inf.check_server():
        print(
            f"[error] Server not reachable at {args.server_url}\n"
            "        Start it with:  python -m alpamayo.server",
            file=sys.stderr,
        )
        sys.exit(1)

    result = inf.run(
        data_dir=args.data_dir,
        log_id=args.log_id,
        task=args.task,
        question=args.question,
        category=args.category,
        stream=args.stream,
    )

    if not args.stream:
        result.display()


if __name__ == "__main__":
    main()
