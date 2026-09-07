#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# JBDetection — Deploy Script
# ═══════════════════════════════════════════════════════════════════════
# Deploys the JBDetection application with the new PaddleOCR pipeline.
#
# Usage:
#   ./deploy.sh              # deploy locally (no Docker)
#   ./deploy.sh --docker     # deploy with Docker Compose
#   ./deploy.sh --gpu        # deploy locally with GPU support
#
# This script:
#   1. Verifies Python version
#   2. Creates a virtual environment
#   3. Installs PaddlePaddle + PaddleOCR (in the correct order)
#   4. Installs all other dependencies
#   5. Runs database migrations (if needed)
#   6. Starts the Gunicorn server
#
# Prerequisites:
#   - Python 3.10+
#   - PostgreSQL 16 (running and accessible)
#   - (Optional) NVIDIA GPU + drivers (for GPU mode)
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
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="${PROJECT_ROOT}/apps/backend"
VENV_DIR="${PROJECT_ROOT}/venv"
REQUIREMENTS="${PROJECT_ROOT}/requirements-linux.txt"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-2}"
THREADS="${THREADS:-4}"

# ── Parse arguments ───────────────────────────────────────────────────
USE_DOCKER=false
USE_GPU=false
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
            echo "  --gpu       Install paddlepaddle-gpu instead of paddlepaddle"
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
    step "Deploying with Docker Compose..."
    cd "${BACKEND_DIR}"
    docker-compose down || true
    docker-compose build
    docker-compose up -d
    info "Docker deployment started."
    info "  App:       http://localhost:${PORT}"
    info "  Test app:  http://localhost:5001"
    info "  Postgres:  localhost:5433"
    info ""
    info "View logs: docker-compose logs -f"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════
# Local deployment path
# ═══════════════════════════════════════════════════════════════════════

# ── Step 1: Check Python version ────────────────────────────────────
step "Step 1/7: Checking Python version..."
if ! command -v python3 &>/dev/null; then
    error "Python 3 is not installed. Install Python 3.10+ first."
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
info "Python version: ${PYTHON_VERSION}"

PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]; }; then
    error "Python 3.10+ is required. Found ${PYTHON_VERSION}"
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

# ── Step 4: Install PaddlePaddle (MUST be before paddleocr) ─────────
step "Step 4/7: Installing PaddlePaddle..."
if [ "$USE_GPU" = true ]; then
    info "Installing paddlepaddle-gpu (GPU mode)..."
    pip install paddlepaddle-gpu==2.6.2
else
    info "Installing paddlepaddle (CPU mode)..."
    pip install paddlepaddle==2.6.2
fi

# ── Step 5: Install PaddleOCR ────────────────────────────────────────
step "Step 5/7: Installing PaddleOCR..."
pip install paddleocr==2.10.0

# ── Step 6: Install remaining dependencies ───────────────────────────
step "Step 6/7: Installing remaining dependencies..."
if [ -f "${REQUIREMENTS}" ]; then
    # Install everything EXCEPT paddlepaddle/paddleocr (already installed above)
    # Use --no-deps for those two to avoid conflicts
    pip install -r "${REQUIREMENTS}" --no-deps paddlepaddle paddleocr || true
    # Then install any missing deps
    pip install -r "${REQUIREMENTS}"
else
    error "Requirements file not found: ${REQUIREMENTS}"
fi

# ── Step 7: Verify installation ──────────────────────────────────────
step "Step 7/7: Verifying installation..."
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

# ── Start the server ─────────────────────────────────────────────────
step "Starting Gunicorn server on port ${PORT}..."
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
