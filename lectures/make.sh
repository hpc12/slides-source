#! /bin/bash
set -e

SOURCE=$1

mkdir -p out

if test "$SOURCE" = ""; then
  echo "usage: $0 source-file-without-extension-and-path"
  exit 1
fi

mv "out/$SOURCE.pdf" "out/$SOURCE-prev.pdf" || true

export TEXINPUTS=$TEXINPUTS:.
export TEXINPUTS=$TEXINPUTS:$SOURCE-media
export TEXINPUTS=$TEXINPUTS:media:slides:code:loopy-examples:../hw
export TEXINPUTS=$TEXINPUTS:$HOME/research/presentations/slides
export TEXINPUTS=$TEXINPUTS:$HOME/research/presentations/slides/snippets
export TEXINPUTS=$TEXINPUTS:$HOME/research/presentations/slides/style
export TEXINPUTS=$TEXINPUTS:$HOME/research/presentations/slides/media

#(cd out; BIBINPUTS=.. bibtex $SOURCE)

pdflatex -output-directory out $SOURCE || ERROR=1
if test "$ERROR" == 1; then
  echo "-----------------------------------------------------------------------"
  echo "An error occurred. Hit enter to see the log."
  echo "-----------------------------------------------------------------------"
  read LINE
  less out/$SOURCE.log
fi
