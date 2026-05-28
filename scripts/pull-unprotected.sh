#!/usr/bin/env bash
set -euo pipefail
# Pulls updates only for files NOT listed in protected-files.txt.
# Usage: bash pull-unprotected.sh [remote] [branch]
# Default: origin devel

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/protected-files.txt"
REMOTE="${1:-origin}"
BRANCH="${2:-devel}"

if [ ! -f "$MANIFEST" ]; then
	echo "Manifest not found: $MANIFEST" >&2
	exit 1
fi

cd "$REPO_ROOT"

# Collect protected entries
PROTECTED=()
while IFS= read -r line; do
	entry="${line%%#*}"
	entry="${entry//$'\r'/}"
	entry="${entry## }"
	entry="${entry%% }"
	[ -z "$entry" ] && continue
	REPO_BASENAME="$(basename "$REPO_ROOT")"
	case "$entry" in
	"$REPO_BASENAME"/*) entry="${entry#"${REPO_BASENAME}"/}" ;;
	esac
	if [ ! -e "$REPO_ROOT/$entry" ] && [[ "$entry" != */* ]]; then
		if [ -e "$REPO_ROOT/config/$entry" ]; then
			entry="config/$entry"
		fi
	fi
	PROTECTED+=("$entry")
done <"$MANIFEST"

# Fetch updates
if ! git fetch "$REMOTE" "$BRANCH"; then
	echo "git fetch failed: $REMOTE $BRANCH" >&2
	exit 2
fi

# Build list of files from remote branch (so new files are included)
mapfile -t TRACKED < <(git ls-tree -r --name-only "$REMOTE/$BRANCH")
UNPROTECTED=()
for f in "${TRACKED[@]}"; do
	skip=false
	for p in "${PROTECTED[@]}"; do
		if [[ "$p" == */ ]]; then
			case "$f" in
			"$p"*)
				skip=true
				break
				;;
			esac
		elif [ "$f" = "$p" ]; then
			skip=true
			break
		fi
	done
	if [ "$skip" = false ]; then
		UNPROTECTED+=("$f")
	fi
done

if [ ${#UNPROTECTED[@]} -eq 0 ]; then
	echo "No unprotected files to update."
	exit 0
fi

# Preserve local permissions (if file exists locally) before checkout
declare -A PERMS
for f in "${UNPROTECTED[@]}"; do
	if [ -f "$REPO_ROOT/$f" ]; then
		PERMS["$f"]=$(stat -c '%a' "$REPO_ROOT/$f" 2>/dev/null || echo "")
	fi
done

# Update only unprotected files from remote branch
while IFS= read -r f; do
	git checkout "$REMOTE/$BRANCH" -- "$f" >/dev/null 2>&1 || true
	if [ -n "${PERMS[$f]:-}" ]; then
		chmod "${PERMS[$f]}" "$REPO_ROOT/$f" 2>/dev/null || true
	fi
	echo "[UPDATED] $f"
done < <(printf "%s\n" "${UNPROTECTED[@]}")

echo "Done. Protected files left untouched."
