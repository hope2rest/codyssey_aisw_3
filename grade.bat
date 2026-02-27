@echo off
if "%~1"=="" (
    python -m grading.scripts.run_submissions
) else (
    python -m grading.scripts.run_submissions --filter %1
)
