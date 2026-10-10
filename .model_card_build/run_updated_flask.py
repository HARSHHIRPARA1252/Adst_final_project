from pathlib import Path
import runpy
import sys
root = Path(r"D:\BGI_Win_Probability")
sys.path.insert(0, str(root / ".venv" / "Lib" / "site-packages"))
sys.path.insert(0, str(root))
runpy.run_path(str(root / "app.py"), run_name="__main__")
