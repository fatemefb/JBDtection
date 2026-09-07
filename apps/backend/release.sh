#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# JBDetection — Release Script
# ═══════════════════════════════════════════════════════════════════════
# Builds a release package for JBDetection with the new PaddleOCR pipeline.
#
# Usage:
#   ./release.sh                          # build release in dist/
#   ./release.sh --version 1.1.0          # specify version
#   ./release.sh --docker                 # build Docker image
#   ./release.sh --clean                  # clean build artifacts first
#
# This script:
#   1. Cleans previous build artifacts
#   2. Runs all tests (unit tests — skips slow E2E)
#   3. Packages the application into a release tarball
#   4. Optionally builds a Docker image
#
# Output:
#   dist/jbdetection-<version>.tar.gz    # Application package
#   dist/jbdetection-<version>.tar.gz.sha256  # Checksum
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Color output ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
VERSION="${VERSION:-1.0.0}"
BUILD_DOCKER=false
CLEAN_FIRST=false
SKIP_TESTS=false

# ── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
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
            echo "Usage: $0 [--version VERSION] [--docker] [--clean] [--skip-tests]"
            echo ""
            echo "Options:"
            echo "  --version VERSION  Set release version (default: 1.0.0)"
            echo "  --docker           Also build Docker image"
            echo "  --clean            Clean build artifacts first"
            echo "  --skip-tests       Skip running tests (not recommended)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            warn "Unknown argument: $1"
            shift
            ;;
    esac
done

info "Building JBDetection release v${VERSION}"

# ── Step 1: Clean previous artifacts ──────────────────────────────────
if [ "$CLEAN_FIRST" = true ]; then
    step "Step 1/5: Cleaning previous build artifacts..."
    rm -rf "${DIST_DIR}"
    find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "${PROJECT_ROOT}" -type f -name "*.pyc" -delete 2>/dev/null || true
    info "Cleaned."
else
    step "Step 1/5: Skipping clean (use --clean to enable)"
fi

# ── Step 2: Run tests ─────────────────────────────────────────────────
if [ "$SKIP_TESTS" = false ]; then
    step "Step 2/5: Running unit tests..."
    cd "${PROJECT_ROOT}"

    # Preload libz for PaddlePaddle
    export LD_PRELOAD="${LD_PRELOAD:-}"
    if [ -f /lib/x86_64-linux-gnu/libz.so.1 ]; then
        export LD_PRELOAD="/lib/x86_64-linux-gnu/libz.so.1:${LD_PRELOAD}"
    fi

    if ! python3 -m pytest scripts/test_jb_detection/ \
            --ignore=scripts/test_jb_detection/test_e2e_paddleocr.py \
            -q --tb=short 2>&1; then
        error "Tests failed. Fix them before building a release."
    fi
    info "All tests passed."
else
    step "Step 2/5: Skipping tests (--skip-tests)"
fi

# ── Step 3: Create dist directory ──────────────────────────────────────
step "Step 3/5: Preparing release package..."
mkdir -p "${DIST_DIR}"

RELEASE_NAME="jbdetection-${VERSION}"
RELEASE_DIR="${DIST_DIR}/${RELEASE_NAME}"
rm -rf "${RELEASE_DIR}"
mkdir -p "${RELEASE_DIR}"

# ── Step 4: Copy files into release package ───────────────────────────
step "Step 4/5: Copying application files..."

# Copy the jb_detection package
mkdir -p "${RELEASE_DIR}/apps/backend/jb_detection"
cp -r "${PROJECT_ROOT}/jb_detection/"*.py "${RELEASE_DIR}/apps/backend/jb_detection/"
cp "${PROJECT_ROOT}/jb_detection/requirements.txt" "${RELEASE_DIR}/apps/backend/jb_detection/"

# Copy backend files (the modified ones)
if [ -d "${PROJECT_ROOT}/modified_files" ]; then
    cp "${PROJECT_ROOT}/modified_files/app.py" "${RELEASE_DIR}/apps/backend/"
    cp "${PROJECT_ROOT}/modified_files/api.py" "${RELEASE_DIR}/apps/backend/"
    cp "${PROJECT_ROOT}/modified_files/TagJBExtractorLogger.py" "${RELEASE_DIR}/apps/backend/"
    cp "${PROJECT_ROOT}/modified_files/LinuxTagJBExtractor.py" "${RELEASE_DIR}/apps/backend/"
    cp "${PROJECT_ROOT}/modified_files/LinuxTagJBExtractorLogger.py" "${RELEASE_DIR}/apps/backend/"
fi

# Copy the requirements file
cp "${PROJECT_ROOT}/deploy/requirements-linux.txt" "${RELEASE_DIR}/"

# Copy the deploy scripts
cp "${PROJECT_ROOT}/deploy/deploy.sh" "${RELEASE_DIR}/"
cp "${PROJECT_ROOT}/deploy/release.sh" "${RELEASE_DIR}/"

# Copy tests
mkdir -p "${RELEASE_DIR}/scripts/test_jb_detection"
cp "${PROJECT_ROOT}/scripts/test_jb_detection/"*.py "${RELEASE_DIR}/scripts/test_jb_detection/"

# Copy docker files if they exist
if [ -f "${PROJECT_ROOT}/repo/JBDtection/apps/backend/Dockerfile" ]; then
    cp "${PROJECT_ROOT}/repo/JBDtection/apps/backend/Dockerfile" "${RELEASE_DIR}/apps/backend/"
fi
if [ -f "${PROJECT_ROOT}/repo/JBDtection/apps/backend/docker-compose.yml" ]; then
    cp "${PROJECT_ROOT}/repo/JBDtection/apps/backend/docker-compose.yml" "${RELEASE_DIR}/apps/backend/"
fi

# Copy database models and services (needed for the app to run)
if [ -d "${PROJECT_ROOT}/repo/JBDtection/apps/backend/db" ]; then
    cp -r "${PROJECT_ROOT}/repo/JBDtection/apps/backend/db" "${RELEASE_DIR}/apps/backend/"
fi
if [ -d "${PROJECT_ROOT}/repo/JBDtection/apps/backend/services" ]; then
    cp -r "${PROJECT_ROOT}/repo/JBDtection/apps/backend/services" "${RELEASE_DIR}/apps/backend/"
fi
if [ -d "${PROJECT_ROOT}/repo/JBDtection/apps/backend/utils" ]; then
    cp -r "${PROJECT_ROOT}/repo/JBDtection/apps/backend/utils" "${RELEASE_DIR}/apps/backend/"
fi
if [ -d "${PROJECT_ROOT}/repo/JBDtection/apps/backend/modules" ]; then
    cp -r "${PROJECT_ROOT}/repo/JBDtection/apps/backend/modules" "${RELEASE_DIR}/apps/backend/"
fi
if [ -d "${PROJECT_ROOT}/repo/JBDtection/apps/backend/frontend" ]; then
    cp -r "${PROJECT_ROOT}/repo/JBDtection/apps/backend/frontend" "${RELEASE_DIR}/apps/backend/"
fi
if [ -f "${PROJECT_ROOT}/repo/JBDtection/apps/backend/pdf_classifier.py" ]; then
    cp "${PROJECT_ROOT}/repo/JBDtection/apps/backend/pdf_classifier.py" "${RELEASE_DIR}/apps/backend/"
fi
if [ -f "${PROJECT_ROOT}/repo/JBDtection/apps/backend/logger_config.py" ]; then
    cp "${PROJECT_ROOT}/repo/JBDtection/apps/backend/logger_config.py" "${RELEASE_DIR}/apps/backend/"
fi
# apps/__init__.py
if [ -f "${PROJECT_ROOT}/repo/JBDtection/apps/__init__.py" ]; then
    mkdir -p "${RELEASE_DIR}/apps"
    cp "${PROJECT_ROOT}/repo/JBDtection/apps/__init__.py" "${RELEASE_DIR}/apps/"
fi

# Remove __pycache__ from the release
find "${RELEASE_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${RELEASE_DIR}" -type f -name "*.pyc" -delete 2>/dev/null || true

# Create a VERSION file
echo "JBDetection ${VERSION}" > "${RELEASE_DIR}/VERSION"
echo "Built: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "${RELEASE_DIR}/VERSION"
echo "Pipeline: PaddleOCR ${PADDLE_VERSION:-2.10.0}" >> "${RELEASE_DIR}/VERSION"

# Create a README
cat > "${RELEASE_DIR}/README.md" << EOF
# JBDetection ${VERSION}

## Quick Start

1. Install dependencies:
   \`\`\`bash
   pip install -r requirements-linux.txt
   \`\`\`

2. Set up the database:
   \`\`\`bash
   export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/jbdetection"
   cd apps/backend && alembic upgrade head
   \`\`\`

3. Run the app:
   \`\`\`bash
   ./deploy.sh
   \`\`\`

Or use Docker:
\`\`\`bash
cd apps/backend && docker-compose up -d
\`\`\`

## Files

- \`apps/backend/jb_detection/\` — The PaddleOCR-based pipeline package
- \`apps/backend/app.py\` — Flask application entry point
- \`apps/backend/api.py\` — REST API blueprint
- \`deploy.sh\` — Local deployment script
- \`requirements-linux.txt\` — Python dependencies (no Tesseract, no TensorFlow)
- \`scripts/test_jb_detection/\` — Test suite

## Removed (vs. legacy)

- Tesseract / pytesseract — replaced by PaddleOCR
- TensorFlow / Keras — optional (PDF classifier is now optional)
- PyTorch — removed (unused)
- RapidFuzz — removed (using python-Levenshtein)

EOF

info "Release package prepared at: ${RELEASE_DIR}"

# ── Step 5: Create tarball ────────────────────────────────────────────
step "Step 5/5: Creating release tarball..."
cd "${DIST_DIR}"
TARBALL="${RELEASE_NAME}.tar.gz"
tar -czf "${TARBALL}" "${RELEASE_NAME}"

# Generate checksum
sha256sum "${TARBALL}" > "${TARBALL}.sha256"

# Get tarball size
SIZE=$(du -h "${TARBALL}" | cut -f1)

info ""
info "╔══════════════════════════════════════════════════════════════╗"
info "║  Release built successfully!                                  ║"
info "╚══════════════════════════════════════════════════════════════╝"
info ""
info "  Version:    ${VERSION}"
info "  Tarball:    ${DIST_DIR}/${TARBALL}"
info "  Size:       ${SIZE}"
info "  Checksum:   ${DIST_DIR}/${TARBALL}.sha256"
info ""

# ── Optional: Build Docker image ──────────────────────────────────────
if [ "$BUILD_DOCKER" = true ]; then
    step "Building Docker image..."
    cd "${RELEASE_DIR}/apps/backend"
    docker build -t "jbdetection:${VERSION}" .
    docker tag "jbdetection:${VERSION}" "jbdetection:latest"
    info "Docker image built: jbdetection:${VERSION}"
fi

# Clean up release directory (keep only tarball)
rm -rf "${RELEASE_DIR}"

info "Done!"
