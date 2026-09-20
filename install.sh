#!/usr/bin/env bash
# bruv verified installer for macOS and Linux.
# Downloads the artifact plus SHA256SUMS, verifies the checksum, then moves the
# binary atomically into a user-owned bin directory. No root required.
set -euo pipefail

DEST_DIR="${XDG_BIN_HOME:-$HOME/.local/bin}"
ARTIFACT_BASE="${BRUV_ARTIFACT_BASE:-https://github.com/bruv/bruv/releases/latest/download}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  arm64|aarch64) ARCH="aarch64" ;;
esac
ARTIFACT="bruv-${OS}-${ARCH}"
CHECKSUMS="SHA256SUMS"

echo "Installing bruv for ${OS}/${ARCH} to ${DEST_DIR}"
mkdir -p "$DEST_DIR"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "Downloading ${ARTIFACT}..."
curl -fsSL "${ARTIFACT_BASE}/${ARTIFACT}" -o "${TMPDIR}/${ARTIFACT}"

echo "Downloading ${CHECKSUMS}..."
curl -fsSL "${ARTIFACT_BASE}/${CHECKSUMS}" -o "${TMPDIR}/${CHECKSUMS}"

# Verify checksum: require the artifact's line to exist and match
LINE="$(grep "${ARTIFACT}" "${TMPDIR}/${CHECKSUMS}" || true)"
if [ -z "$LINE" ]; then
  echo "error: ${ARTIFACT} missing from ${CHECKSUMS}" >&2
  exit 1
fi
EXPECTED="$(echo "$LINE" | awk '{print $1}')"
ACTUAL="$(sha256sum "${TMPDIR}/${ARTIFACT}" | awk '{print $1}')"
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "error: checksum mismatch for ${ARTIFACT}" >&2
  echo "  expected: ${EXPECTED}" >&2
  echo "  actual:   ${ACTUAL}" >&2
  exit 1
fi
echo "Checksum verified."

chmod +x "${TMPDIR}/${ARTIFACT}"
mv "${TMPDIR}/${ARTIFACT}" "${DEST_DIR}/bruv"

echo "Installed: ${DEST_DIR}/bruv"
echo "Add ${DEST_DIR} to your PATH if it is not already there:"
echo "  export PATH=\"${DEST_DIR}:\$PATH\""
