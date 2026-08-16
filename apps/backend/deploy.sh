#!/bin/bash

set -e

VERSION="$1"

if [ -z "$VERSION" ]; then
    echo "Usage: ./apps/backend/deploy.sh <version>"
    echo "Example: ./apps/backend/deploy.sh 1.1.0"
    exit 1
fi

IMAGE="jbdetection-image-v1:${VERSION}"

echo "======================================"
echo " JBDetection Deployment"
echo " Version: ${VERSION}"
echo " Image:   ${IMAGE}"
echo "======================================"

# Check that requested release exists
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo ""
    echo "ERROR: Image ${IMAGE} does not exist."
    echo "Deploy aborted."
    exit 1
fi

echo ""
echo "Image found."

docker image inspect "$IMAGE" \
    --format 'IMAGE={{.RepoTags}} ID={{.Id}}'

echo ""
echo "Deploying ${VERSION}..."

IMAGE_TAG="$VERSION" docker compose \
    -f apps/backend/docker-compose.yml \
    up -d \
    --no-build \
    jbdetection_v1

echo ""
echo "Verifying running container..."

docker inspect jbdetection_v1 \
    --format 'Container={{.Name}} Image={{.Config.Image}}'

echo ""
echo "Deployment ${VERSION} completed."
