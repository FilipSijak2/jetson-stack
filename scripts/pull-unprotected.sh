#!/usr/bin/env bash
set -euo pipefail

# Pulls only runtime stack files while preserving local protected config.
# Usage: bash pull-unprotected.sh [remote] [branch] [runtime-files-manifest]
# Defaults: origin devel scripts/runtime-files.txt

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROTECTED_MANIFEST="$SCRIPT_DIR/protected-files.txt"
RUNTIME_MANIFEST="${3:-$SCRIPT_DIR/runtime-files.txt}"
REMOTE="${1:-origin}"
BRANCH="${2:-devel}"

if [ ! -f "$PROTECTED_MANIFEST" ]; then
	echo "Protected manifest not found: $PROTECTED_MANIFEST" >&2
	exit 1
fi

cd "$REPO_ROOT"

normalize_entry() {
	local entry="$1"
	entry="${entry%%#*}"
	entry="${entry//$'\r'/}"
	entry="${entry#"${entry%%[![:space:]]*}"}"
	entry="${entry%"${entry##*[![:space:]]}"}"
	local repo_basename
	repo_basename="$(basename "$REPO_ROOT")"
	case "$entry" in
	"$repo_basename"/*) entry="${entry#"$repo_basename"/}" ;;
	esac
	printf '%s\n' "$entry"
}

read_manifest() {
	local manifest="$1"
	local -n output="$2"
	output=()
	while IFS= read -r raw_line || [ -n "$raw_line" ]; do
		local entry
		entry="$(normalize_entry "$raw_line")"
		[ -z "$entry" ] && continue
		output+=("$entry")
	done <"$manifest"
}

runtime_candidate_matches() {
	local file="$1"
	local candidate="$2"
	if [[ "$candidate" == */ ]]; then
		case "$file" in "$candidate"*) return 0 ;; esac
	elif [ "$file" = "$candidate" ]; then
		return 0
	fi
	return 1
}

is_protected_existing_path() {
	local file="$1"
	local protected="$2"
	if [[ "$protected" == */ ]]; then
		case "$file" in
		"$protected"*)
			[ -e "$REPO_ROOT/$protected" ]
			return
			;;
		esac
	elif [ "$file" = "$protected" ]; then
		[ -e "$REPO_ROOT/$protected" ]
		return
	fi
	return 1
}

PROTECTED_ENTRIES=()
read_manifest "$PROTECTED_MANIFEST" PROTECTED_ENTRIES

RUNTIME_ENTRIES=()
if [ -f "$RUNTIME_MANIFEST" ]; then
	read_manifest "$RUNTIME_MANIFEST" RUNTIME_ENTRIES
else
	echo "[WARN] Runtime manifest not found: $RUNTIME_MANIFEST; falling back to all tracked files." >&2
fi

if ! git fetch "$REMOTE" "$BRANCH"; then
	echo "git fetch failed: $REMOTE $BRANCH" >&2
	echo "Hint: if this is a private HTTPS remote, configure Git credentials on Jetson or run via the GitHub Actions deploy workflow." >&2
	exit 2
fi

mapfile -t TRACKED < <(git ls-tree -r --name-only "$REMOTE/$BRANCH")
CANDIDATES=()
for file in "${TRACKED[@]}"; do
	if [ ${#RUNTIME_ENTRIES[@]} -eq 0 ]; then
		CANDIDATES+=("$file")
		continue
	fi
	for runtime_entry in "${RUNTIME_ENTRIES[@]}"; do
		if runtime_candidate_matches "$file" "$runtime_entry"; then
			CANDIDATES+=("$file")
			break
		fi
	done
done

UNPROTECTED=()
for file in "${CANDIDATES[@]}"; do
	skip=false
	for protected_entry in "${PROTECTED_ENTRIES[@]}"; do
		if is_protected_existing_path "$file" "$protected_entry"; then
			skip=true
			break
		fi
	done
	if [ "$skip" = false ]; then
		UNPROTECTED+=("$file")
	fi
done

if [ ${#UNPROTECTED[@]} -eq 0 ]; then
	echo "No unprotected runtime files to update."
	exit 0
fi

echo "Updating ${#UNPROTECTED[@]} runtime file(s) from $REMOTE/$BRANCH"

declare -A PERMS
for file in "${UNPROTECTED[@]}"; do
	if [ -f "$REPO_ROOT/$file" ]; then
		PERMS["$file"]=$(stat -c '%a' "$REPO_ROOT/$file" 2>/dev/null || echo "")
	fi
done

while IFS= read -r file; do
	git checkout "$REMOTE/$BRANCH" -- "$file" >/dev/null 2>&1 || true
	if [ -n "${PERMS[$file]:-}" ]; then
		chmod "${PERMS[$file]}" "$REPO_ROOT/$file" 2>/dev/null || true
	fi
	echo "[UPDATED] $file"
done < <(printf "%s\n" "${UNPROTECTED[@]}")

echo "Done. Protected local files left untouched."
