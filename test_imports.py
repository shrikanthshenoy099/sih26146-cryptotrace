"""
Run this after `pip install -r requirements.txt` to confirm every
teammate's environment is set up correctly before Phase 1 begins.
"""
import importlib

packages = [
    "pandas", "numpy", "geoip2", "torch", "torch_geometric",
    "networkx", "fastapi", "streamlit", "pyvis", "sklearn",
]

failed = []
for pkg in packages:
    try:
        importlib.import_module(pkg)
        print(f"[OK]   {pkg}")
    except ImportError as e:
        print(f"[FAIL] {pkg}  ->  {e}")
        failed.append(pkg)

if failed:
    print(f"\n{len(failed)} package(s) failed to import: {failed}")
    print("Fix these before starting Phase 1.")
else:
    print("\nAll imports OK — environment is ready.")
