#!/usr/bin/sh
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
biblatex main.aux
lualatex -shell-escape main.tex
