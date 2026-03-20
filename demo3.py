"""Legacy compatibility wrapper.

This file is kept so existing usage of `python demo3.py ...` still works.
The implementation now lives in `legacy_demo_variant.py`.
"""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("legacy_demo_variant.py")), run_name="__main__")
