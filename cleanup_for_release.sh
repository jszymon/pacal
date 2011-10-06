#!/bin/sh

# remove files not needed in a release

find . -name '*.pyc' -print -delete  # .pyc files
find . -name '*~' -print -delete     # Emacs backups
find . -name '*.so' -print -delete   # compiled Cython libraries
