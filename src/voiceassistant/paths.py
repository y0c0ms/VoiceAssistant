"""paths.py: where the code finds its own files and the local resources it runs with.

The package holds the code and the example config. The repository root holds what is not code and not committed:
the speech models in voices/ and the virtualenv for the Portuguese voice in .venv-phoonnx.
"""
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, "..", ".."))
EXAMPLE_CONFIG = os.path.join(PACKAGE_DIR, "config.example.yaml")
VOICES_DIR = os.path.join(REPO_ROOT, "voices")
PHOONNX_PYTHON = os.path.join(REPO_ROOT, ".venv-phoonnx", "bin", "python")
TUGAPHONE_SCRIPT = os.path.join(PACKAGE_DIR, "speech", "tts_tugaphone.py")
