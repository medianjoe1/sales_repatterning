import os
import sys
import subprocess

# Launch the app using the current Python interpreter
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
