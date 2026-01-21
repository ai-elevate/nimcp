#!/bin/bash
# Script to add exception macros include to all C files in src/core/brain
# that don't already have it

cd /home/bbrelin/nimcp
count=0

for f in $(find src/core/brain -name "*.c" | xargs grep -L "nimcp_exception_macros\.h" 2>/dev/null); do
  # Get line number of last quoted include (nimcp-style include)
  last_line=$(grep -n '#include "' "$f" 2>/dev/null | tail -1 | cut -d: -f1)

  if [ -n "$last_line" ]; then
    # Insert the include after the last quoted include
    sed -i "${last_line}a\\#include \"utils/exception/nimcp_exception_macros.h\"" "$f"
    echo "Updated: $f"
    count=$((count+1))
  else
    echo "Skipped (no quoted includes): $f"
  fi
done

echo ""
echo "Total files updated: $count"
