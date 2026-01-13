#!/usr/bin/env bash
# Setup script for andweorc git hooks
#
# This script configures git to use the project's git hooks directory.
# Run once after cloning the repository:
#
#   ./.githooks/setup.sh
#
# The hooks will then be active for all git operations.

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Setting up git hooks for andweorc..."

# Configure git to use the .githooks directory
git config core.hooksPath "$SCRIPT_DIR"

echo "Git hooks configured!"
echo ""
echo "Active hooks:"
for hook in "$SCRIPT_DIR"/*; do
    if [ -x "$hook" ] && [ "$(basename "$hook")" != "setup.sh" ]; then
        echo "  - $(basename "$hook")"
    fi
done
echo ""
echo "To disable hooks temporarily, use:"
echo "  git commit --no-verify"
echo "  git push --no-verify"
