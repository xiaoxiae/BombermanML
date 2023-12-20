#!/usr/bin/sh
lualatex -shell-escape main.tex
lualatex -shell-escape main.tex
bibtex main.aux
lualatex -shell-escape main.tex
