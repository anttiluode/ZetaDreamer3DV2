@echo off
cd /d "%~dp0"
python3.13 zeta_dreamer_3d.py
if errorlevel 1 (
  echo.
  echo Python 3.13 or a dependency is missing.
  echo Try: python3.13 -m pip install numpy torch Pillow opencv-python
  pause
)
