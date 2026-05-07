from __future__ import annotations

import os
import shutil
import subprocess


def run(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return f"{command[0]}: not found"
    output = (completed.stdout + completed.stderr).strip()
    return output or f"{command[0]} exited with {completed.returncode}"


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    print("nvidia-smi:")
    print(run(["nvidia-smi"]))
    print()

    print("NVIDIA PCI devices:")
    lspci = shutil.which("lspci")
    if lspci:
        lines = [line for line in run([lspci]).splitlines() if "nvidia" in line.lower()]
        print("\n".join(lines) if lines else "No NVIDIA devices found by lspci.")
    else:
        print("lspci: not found")
    print()

    import tensorflow as tf

    print(f"TensorFlow: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")


if __name__ == "__main__":
    main()
