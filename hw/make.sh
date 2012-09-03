#! /bin/sh
mkdir -p out
TEXINPUTS=:.. pdflatex -output-directory=out "$1"
