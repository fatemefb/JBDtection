#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# JBDetection — Deploy Script
# ═══════════════════════════════════════════════════════════════════════
# Deploys the JBDetection application with the new PaddleOCR pipeline.
#
# Usage:
#   ./deploy.sh              # deploy locally (no Docker)
#   ./deploy.sh --docker     # deploy with Docker Compose
#   ./deploy.sh --gpu        # deploy locally with the validated K80 GPU build
#
# This script:
#   1. Verifies Python version
#   2. Creates a virtual environment
#   3. Installs the validated custom PaddlePaddle GPU wheel
#   4. Installs PaddleOCR and all other dependencies
#   5. Verifies the runtime installation
#   6. Runs database migrations (if needed)
#   7. Starts the Gunicorn server
#
# Prerequisites:
#   - Python 3.8
#   - PostgreSQL 16 (running and accessible)
#   - NVIDIA GPU + drivers compatible with the validated K80 build
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Color output ─────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "${BLUE}[STEP]${NC}  $*"; }

# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/apps/backend"
VENV_DIR="${PROJECT_ROOT}/venv"
REQUIREMENTS="${PROJECT_ROOT}/requirements-linux.txt"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"

# ── Parse arguments ───────────────────────────────────────────────────
USE_DOCKER=false
USE_GPU=true
SKIP_VENV=false

for arg in "$@"; do
    case $arg in
        --docker)  USE_DOCKER=true ;;
        --gpu)     USE_GPU=true ;;
        --no-venv) SKIP_VENV=true ;;
        --help|-h)
            echo "Usage: $0 [--docker] [--gpu] [--no-venv]"
            echo ""
            echo "Options:"
            echo "  --docker    Deploy using docker-compose"
            echo "  --gpu       Use the validated custom PaddlePaddle GPU build (default)"
            echo "  --no-venv   Skip virtual environment creation"
            echo "  --help      Show this help"
            exit 0
            ;;
        *)
            warn "Unknown argument: $arg"
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
# Docker deployment path
# ═══════════════════════════════════════════════════════════════════════
if [ "$USE_DOCKER" = true ]; then
    step "Deploying jbdetection_test with Docker Compose..."
    cd "${BACKEND_DIR}"

    # IMPORTANT:
    # This deployment path intentionally targets ONLY jbdetection_test.
    # jbdetection_v1 must remain untouched.
    docker compose build jbdetection_test
    docker compose up -d --no-deps jbdetection_test

    info "Docker test deployment started."
    info "  Test app:  http://localhost:5001"
    info "  V1:        untouched"
    info ""
    info "View test logs: docker compose logs -f jbdetection_test"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════
# Local deployment path
# ═══════════════════════════════════════════════════════════════════════

# ── Step 1: Check Python version ────────────────────────────────────
step "Step 1/7: Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed. Python 3.8 is required."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python version: ${PYTHON_VERSION}"

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -ne 8 ]; then
    error "Python 3.8 is required for the validated custom PaddlePaddle wheel. Found ${PYTHON_VERSION}"
fi

# ── Step 2: Create virtual environment ───────────────────────────────
if [ "$SKIP_VENV" = false ]; then
    step "Step 2/7: Creating virtual environment..."
    if [ ! -d "${VENV_DIR}" ]; then
        python3 -m venv "${VENV_DIR}"
        info "Virtual environment created at: ${VENV_DIR}"
    else
        info "Virtual environment already exists at: ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
else
    step "Step 2/7: Skipping virtual environment (--no-venv)"
fi

# ── Step 3: Upgrade pip ──────────────────────────────────────────────
step "Step 3/7: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# ── Step 4: Install validated custom PaddlePaddle GPU build ──────────
step "Step 4/7: Installing validated PaddlePaddle GPU build..."

PADDLE_WHEEL="${PROJECT_ROOT}/apps/backend/build_artifacts/paddle/paddlepaddle_gpu-0.0.0-cp38-cp38-linux_x86_64.whl"

if [ ! -f "${PADDLE_WHEEL}" ]; then
    error "Custom PaddlePaddle wheel not found: ${PADDLE_WHEEL}"
fi

info "Installing custom PaddlePaddle wheel for Tesla K80..."
pip install --no-cache-dir "${PADDLE_WHEEL}"

# ── Step 5: Install PaddleOCR and dependencies ───────────────────────
step "Step 5/7: Installing PaddleOCR and runtime dependencies..."
if [ -f "${REQUIREMENTS}" ]; then
    pip install --no-cache-dir -r "${REQUIREMENTS}"
else
    error "Requirements file not found: ${REQUIREMENTS}"
fi

# ── Step 6: Verify installation ──────────────────────────────────────
step "Step 6/7: Verifying installation..."
python3 -c "
import ctypes
ctypes.CDLL('libz.so.1', mode=ctypes.RTLD_GLOBAL)
import paddle
print(f'  paddle: {paddle.__version__} (CUDA: {paddle.is_compiled_with_cuda()})')
from paddleocr import PaddleOCR
print('  paddleocr: imported')
import cv2, numpy, fitz
print(f'  cv2: {cv2.__version__}, numpy: {numpy.__version__}, fitz: {fitz.VersionBind}')
print()
print('✓ All dependencies installed successfully!')
"

# ── Run database migrations ──────────────────────────────────────────
if [ -n "${DATABASE_URL:-}" ]; then
    step "Running database migrations..."
    cd "${BACKEND_DIR}"
    alembic upgrade head
    info "Migrations complete."
else
    warn "DATABASE_URL not set — skipping migrations."
    warn "Set it with: export DATABASE_URL='postgresql+psycopg2://user:pass@host:port/dbname'"
fi

# ── Step 7: Start the server ─────────────────────────────────────────
step "Step 7/7: Starting Gunicorn server on port ${PORT}..."
cd "${BACKEND_DIR}"
info "  Workers: ${WORKERS}"
info "  Threads: ${THREADS}"
info "  URL:     http://localhost:${PORT}"
info ""
info "Press Ctrl+C to stop."

exec gunicorn apps.backend.app:app \
    -b "0.0.0.0:${PORT}" \
    --workers="${WORKERS}" \
    --threads="${THREADS}" \
    --timeout=3600 \
    --graceful-timeout=3600 \
    --keep-alive=5 \
    --max-requests=500 \
    --max-requests-jitter=50 \
    --worker-class=sync \
    --log-level=info \
    --access-logfile=- \
    --error-logfile=-
