import os
import sys
import runpy

# Ensure smell-ai directory is in path and working context
current_dir = os.path.dirname(os.path.abspath(__file__))
smell_ai_dir = os.path.join(current_dir, "smell-ai")

if smell_ai_dir not in sys.path:
    sys.path.insert(0, smell_ai_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Execute the application
app_path = os.path.join(smell_ai_dir, "app.py")
runpy.run_path(app_path, run_name="__main__")
