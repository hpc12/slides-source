#! /bin/bash

set -e

SOURCE=$1

if test "$SOURCE" = ""; then
  echo "usage: $0 source-file-without-extension-and-path"
  exit 1
fi

mkdir -p out

mv "out/$SOURCE.pdf" "out/$SOURCE-prev.pdf" || true

export BIBINPUTS=..
(cd out; bibtex $SOURCE) || true

export TEXINPUTS=$TEXINPUTS:.:./packages/algorithmic:./packages/elsarticle:./data/out:./out:./media
pdflatex -output-directory=out "$1" || ERROR=1

if test "$ERROR" == 1; then
  echo "-----------------------------------------------------------------------"
  echo "An error occurred. Hit enter to see the log."
  echo "-----------------------------------------------------------------------"
  read LINE
  less out/$SOURCE.log
fi
