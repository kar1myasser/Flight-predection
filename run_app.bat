@echo off
cd /d "%~dp0"
call "C:\Users\kar1m\Desktop\Workspace\my_env\Scripts\activate.bat"
python -m streamlit run app.py
pause
