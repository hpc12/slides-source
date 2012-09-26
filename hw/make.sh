#! /bin/sh
mkdir -p out
TEXINPUTS=:media pdflatex -output-directory=out "$1"
