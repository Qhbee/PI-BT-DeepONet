@echo off
REM 阶段6 正式实验：路线A -> 路线B，hard vs multi-CLS，15 epochs
REM 用法: run_stage6.bat [epochs]
set EPOCHS=15
if not "%~1"=="" set EPOCHS=%~1

cd /d "%~dp0\.."
echo ========== 路线 A：参数化输入 (Kovasznay + Beltrami) ==========
python scripts/run_stage6_experiments.py --route A --branch both --epochs %EPOCHS%

echo.
echo ========== 路线 B：边界/初值序列输入 (Kovasznay + Beltrami) ==========
python scripts/run_stage6_experiments.py --route B --branch both --epochs %EPOCHS%

echo.
echo ========== 全部完成 ==========
pause
