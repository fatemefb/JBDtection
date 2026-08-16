#!/bin/bash
set -e
VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: ./apps/backend/release.sh <version>"
    echo "Example: ./apps/backend/release.sh 1.1.0"
    exit 1
fi

IMAGE="jbdetection-image-v1:${VERSION}"

echo "======================================"
echo " JBDetection Release"
echo " Version: ${VERSION}"
echo " Image:   ${IMAGE}"
echo "======================================"

# Prevent accidental overwrite
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Image ${IMAGE} already exists."
    echo "Release aborted to protect the existing version."
    exit 1
fi

echo ""
echo "Building ${IMAGE}..."

IMAGE_TAG="$VERSION" docker compose \
    -f apps/backend/docker-compose.yml \
    build jbdetection_v1

echo ""
echo "Verifying image..."

docker image inspect "$IMAGE" \
    --format 'IMAGE={{.RepoTags}} ID={{.Id}}'

echo ""
echo "Release ${VERSION} created successfully."