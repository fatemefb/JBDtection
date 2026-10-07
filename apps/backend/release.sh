#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# JBDetection — Release Script
# ═══════════════════════════════════════════════════════════════════════
# Creates a self-contained JBDetection release package.
#
# Usage:
#   ./apps/backend/release.sh
#   ./apps/backend/release.sh --version 1.1.0
#   ./apps/backend/release.sh --clean
#   ./apps/backend/release.sh --skip-tests
#   ./apps/backend/release.sh --docker
#
# The release preserves the repository layout because the Dockerfile
# expects the repository root as its build context.
#
# GPU runtime:
#   PaddleOCR 2.10.0
#   Custom PaddlePaddle GPU build from Paddle 2.4.2 source
#   CUDA 11.4 / sm_37 / NVIDIA Tesla K80
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

# ── Repository paths ──────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/apps/backend"
DIST_DIR="${PROJECT_ROOT}/dist"

VERSION="${VERSION:-1.0.0}"
BUILD_DOCKER=false
CLEAN_FIRST=false
SKIP_TESTS=false

# ── Required files ────────────────────────────────────────────────────
PADDLE_WHEEL="${BACKEND_DIR}/build_artifacts/paddle/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"

for required in \
    "${BACKEND_DIR}/Dockerfile" \
    "${BACKEND_DIR}/docker-compose.yml" \
    "${BACKEND_DIR}/deploy.sh" \
    "${BACKEND_DIR}/requirements-linux.txt" \
    "${BACKEND_DIR}/jb_detection/requirements.txt" \
    "${PADDLE_WHEEL}" \
    "${PROJECT_ROOT}/requirements-linux.txt"
do
    [ -f "${required}" ] || error "Required file not found: ${required}"
done

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || error "--version requires a value"
            VERSION="$2"
            shift 2
            ;;
        --docker)
            BUILD_DOCKER=true
            shift
            ;;
        --clean)
            CLEAN_FIRST=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help|-h)
            cat <<HELP
Usage: $0 [--version VERSION] [--docker] [--clean] [--skip-tests]

Options:
  --version VERSION  Set release version (default: 1.0.0)
  --docker           Also build a standalone Docker image
  --clean            Remove previous dist/ before packaging
  --skip-tests       Skip release test suite
  --help             Show this help
HELP
            exit 0
            ;;
        *)
            warn "Unknown argument: $1"
            shift
            ;;
    esac
done

info "Building JBDetection release v${VERSION}"
info "Repository root: ${PROJECT_ROOT}"

# ── Step 1: Clean ─────────────────────────────────────────────────────
if [ "${CLEAN_FIRST}" = true ]; then
    step "Step 1/5: Cleaning previous release artifacts..."
    rm -rf "${DIST_DIR}"
    find "${PROJECT_ROOT}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete 2>/dev/null || true
    info "Cleaned."
else
    step "Step 1/5: Skipping clean (use --clean to enable)"
fi

# ── Step 2: Tests ─────────────────────────────────────────────────────
if [ "${SKIP_TESTS}" = false ]; then
    step "Step 2/5: Running available tests..."
    cd "${PROJECT_ROOT}"

    # PaddlePaddle / CUDA compatibility
    if [ -f /lib/x86_64-linux-gnu/libz.so.1 ]; then
        export LD_PRELOAD="/lib/x86_64-linux-gnu/libz.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
    fi

    if ! python3 -m pytest \
        apps/backend/tests \
        scripts \
        -q \
        --tb=short \
        --ignore=scripts/test_scanned_pdf.py \
        2>&1
    then
        error "Tests failed. Fix them before building a release."
    fi

    info "Available release tests passed."
else
    step "Step 2/5: Skipping tests (--skip-tests)"
fi

# ── Step 3: Prepare release directory ─────────────────────────────────
step "Step 3/5: Preparing release package..."

mkdir -p "${DIST_DIR}"

RELEASE_NAME="jbdetection-${VERSION}"
RELEASE_DIR="${DIST_DIR}/${RELEASE_NAME}"

rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"

# ── Step 4: Copy current repository runtime ───────────────────────────
step "Step 4/5: Copying application files..."

cd "${PROJECT_ROOT}"

# Preserve the repository layout required by Dockerfile and imports.
mkdir -p "${RELEASE_DIR}/apps"
cp -a "${PROJECT_ROOT}/apps/__init__.py" "${RELEASE_DIR}/apps/"

# Copy backend source tree.
cp -a "${BACKEND_DIR}" "${RELEASE_DIR}/apps/"

# Copy root runtime requirements.
cp "${PROJECT_ROOT}/requirements-linux.txt" "${RELEASE_DIR}/"

# Copy release/deployment scripts explicitly.
cp "${BACKEND_DIR}/deploy.sh" "${RELEASE_DIR}/deploy.sh"
cp "${BACKEND_DIR}/release.sh" "${RELEASE_DIR}/release.sh"

# Copy the current test scripts.
mkdir -p "${RELEASE_DIR}/scripts"
cp -a "${PROJECT_ROOT}/scripts/." "${RELEASE_DIR}/scripts/"

# Remove runtime/build data that should never ship.
rm -rf \
    "${RELEASE_DIR}/apps/backend/outputs_v1" \
    "${RELEASE_DIR}/apps/backend/outputs_v2" \
    "${RELEASE_DIR}/apps/backend/outputs_test" \
    "${RELEASE_DIR}/apps/backend/base_outputs" \
    "${RELEASE_DIR}/apps/backend/base_logs" \
    "${RELEASE_DIR}/apps/backend/logs_v1" \
    "${RELEASE_DIR}/apps/backend/logs_v2" \
    "${RELEASE_DIR}/apps/backend/logs_test"

# Remove Python caches.
find "${RELEASE_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "${RELEASE_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true

# Create VERSION metadata.
cat > "${RELEASE_DIR}/VERSION" <<VERSION_EOF
JBDetection ${VERSION}
Built: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Pipeline: PaddleOCR 2.10.0
PaddlePaddle: custom build from Paddle 2.4.2 source
CUDA: 11.4
GPU architecture: sm_37
Target GPU: NVIDIA Tesla K80
VERSION_EOF

# Create release README reflecting the actual current state.
cat > "${RELEASE_DIR}/README.md" <<README_EOF
# JBDetection ${VERSION}

## Runtime

- Python 3.8
- PaddleOCR 2.10.0
- Custom PaddlePaddle GPU build from Paddle 2.4.2 source
- CUDA 11.4
- GPU architecture: sm_37
- Target GPU: NVIDIA Tesla K80
- PyMuPDF-based digital PDF extraction
- PaddleOCR-based OCR pipeline

The custom PaddlePaddle wheel is included at:

\`apps/backend/build_artifacts/paddle/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl\`

Do not replace this wheel with the standard PaddlePaddle GPU wheel.
The custom build is required for Tesla K80 / sm_37 compatibility.

## Quick Start

Install runtime dependencies:

\`\`\`bash
pip install -r requirements-linux.txt
\`\`\`

Run the local deployment script:

\`\`\`bash
./deploy.sh
\`\`\`

For database migration:

\`\`\`bash
export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/jbdetection"
cd apps/backend
alembic upgrade head
\`\`\`

## Docker

The Dockerfile must be built using the release root as the build context:

\`\`\`bash
docker build -f apps/backend/Dockerfile -t jbdetection:${VERSION} .
\`\`\`

For Compose:

\`\`\`bash
docker compose -f apps/backend/docker-compose.yml up -d jbdetection_test
\`\`\`

Do not rebuild or recreate the \`jbdetection_v1\` service unless explicitly intended.

## Repository Layout

- \`apps/backend/jb_detection/\` — Current JBDetection pipeline
- \`apps/backend/app.py\` — Flask application
- \`apps/backend/api.py\` — REST API
- \`apps/backend/Dockerfile\` — Production/test container image
- \`apps/backend/docker-compose.yml\` — Docker Compose configuration
- \`apps/backend/deploy.sh\` — Deployment helper
- \`apps/backend/release.sh\` — Release packaging
- \`apps/backend/tests/\` — Backend tests
- \`scripts/\` — Project-level tests
- \`requirements-linux.txt\` — Local Linux runtime requirements

## Current Migration State

The active JBDetection OCR pipeline uses PaddleOCR.

Legacy compatibility files may still be present in the backend source tree while the remaining legacy classifier path is being retired. They are intentionally preserved in this release until that migration is fully completed.

RapidFuzz remains part of the current runtime requirements.
README_EOF

info "Release package prepared at: ${RELEASE_DIR}"

# ── Step 5: Tarball ───────────────────────────────────────────────────
step "Step 5/5: Creating release tarball..."

cd "${DIST_DIR}"

TARBALL="${RELEASE_NAME}.tar.gz"

rm -f "${TARBALL}" "${TARBALL}.sha256"

tar -czf "${TARBALL}" "${RELEASE_NAME}"
sha256sum "${TARBALL}" > "${TARBALL}.sha256"

SIZE="$(du -h "${TARBALL}" | cut -f1)"

info ""
info "╔══════════════════════════════════════════════════════════════╗"
info "║  Release built successfully!                                ║"
info "╚══════════════════════════════════════════════════════════════╝"
info ""
info "  Version:    ${VERSION}"
info "  Tarball:    ${DIST_DIR}/${TARBALL}"
info "  Size:       ${SIZE}"
info "  Checksum:   ${DIST_DIR}/${TARBALL}.sha256"
info ""

# ── Optional standalone Docker build ──────────────────────────────────
if [ "${BUILD_DOCKER}" = true ]; then
    step "Building standalone Docker image from release context..."

    cd "${RELEASE_DIR}"

    docker build \
        -f apps/backend/Dockerfile \
        -t "jbdetection:${VERSION}" \
        .

    docker tag "jbdetection:${VERSION}" "jbdetection:latest"

    info "Docker image built: jbdetection:${VERSION}"
fi

# Keep only distributable files.
rm -rf "${RELEASE_DIR}"

info "Done."
