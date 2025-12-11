#!/bin/bash
# Add global_workspace_immune.c to CMakeLists.txt

FILE="src/lib/CMakeLists.txt"
LINE="    \${CMAKE_CURRENT_SOURCE_DIR}/../cognitive/global_workspace/nimcp_global_workspace_immune.c  # Global Workspace - Immune System Integration"

# Check if line already exists
if grep -q "nimcp_global_workspace_immune.c" "$FILE"; then
    echo "Line already exists in $FILE"
    exit 0
fi

# Add line after nimcp_global_workspace_shannon.c
sed -i '/nimcp_global_workspace_shannon.c/a\'"$LINE" "$FILE"

echo "Added nimcp_global_workspace_immune.c to $FILE"
