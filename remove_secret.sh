#!/bin/bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch src/ui/app.py" \
  --prune-empty --tag-name-filter cat -- --all
