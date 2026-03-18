@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Building main.pdf with xelatex...
xelatex -interaction=nonstopmode main.tex
echo.
echo Running bibtex...
bibtex main
echo.
echo Running xelatex (2nd pass)...
xelatex -interaction=nonstopmode main.tex
echo.
echo Running xelatex (3rd pass)...
xelatex -interaction=nonstopmode main.tex
echo.
echo Done. Output: main.pdf
