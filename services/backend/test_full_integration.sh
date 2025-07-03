#!/bin/bash
# Test full integration: Upload image and check detection results

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}===================================${NC}"
echo -e "${BLUE}Full Integration Test${NC}"
echo -e "${BLUE}===================================${NC}"

# Backend API URL
API_URL="http://localhost:8000"
TEST_IMAGE="data/test_coin.jpg"

# Check if test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    echo -e "${RED}Error: Test image not found at $TEST_IMAGE${NC}"
    echo -e "${YELLOW}Please ensure the test image is copied to the data directory${NC}"
    exit 1
fi

# Check if backend is running
echo -e "${BLUE}Checking backend service...${NC}"
if ! curl -sf "${API_URL}/health" -o /dev/null; then
    echo -e "${RED}❌ Backend service is not running${NC}"
    echo -e "${YELLOW}Please start the backend service first:${NC}"
    echo -e "${BLUE}./start-dev-real.sh${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Backend service is running${NC}"

# Check configuration
echo -e "${BLUE}\nChecking backend configuration...${NC}"
CONFIG=$(curl -s "${API_URL}/config")
echo -e "${YELLOW}Configuration:${NC}"
echo "$CONFIG" | jq '.'

# Check if MODE is real
MODE=$(echo "$CONFIG" | jq -r '.mode')
if [ "$MODE" != "real" ]; then
    echo -e "${RED}❌ Backend is not running in REAL mode (current: $MODE)${NC}"
    echo -e "${YELLOW}Please restart backend with MODE=real${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Backend is running in REAL mode${NC}"

# Upload image
echo -e "${BLUE}\nUploading test image...${NC}"
UPLOAD_RESPONSE=$(curl -s -X POST "${API_URL}/images/" \
    -F "file=@${TEST_IMAGE}")

echo -e "${YELLOW}Upload response:${NC}"
echo "$UPLOAD_RESPONSE" | jq '.'

# Check if it's an error response
ERROR_DETAIL=$(echo "$UPLOAD_RESPONSE" | jq -r '.detail // empty')
if [[ "$ERROR_DETAIL" == *"already exists"* ]]; then
    # Extract image ID from error message
    IMAGE_ID=$(echo "$ERROR_DETAIL" | grep -oE '[a-f0-9]{64}')
    echo -e "${YELLOW}⚠️  Image already exists with ID: $IMAGE_ID${NC}"
    echo -e "${BLUE}Using existing image for testing...${NC}"
else
    # Extract image ID from successful upload
    IMAGE_ID=$(echo "$UPLOAD_RESPONSE" | jq -r '.image_id // empty')
    if [ "$IMAGE_ID" == "null" ] || [ -z "$IMAGE_ID" ]; then
        echo -e "${RED}❌ Failed to upload image${NC}"
        echo -e "${RED}Response: $UPLOAD_RESPONSE${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Image uploaded successfully with ID: $IMAGE_ID${NC}"
    
    # Check if there were detections
    DETECTION_COUNT=$(echo "$UPLOAD_RESPONSE" | jq -r '.detection.count // 0')
    if [ "$DETECTION_COUNT" -gt 0 ]; then
        echo -e "${GREEN}✅ Detected $DETECTION_COUNT objects during upload!${NC}"
        echo -e "${YELLOW}\nDetections:${NC}"
        echo "$UPLOAD_RESPONSE" | jq '.detection.detections[]'
    fi
fi

# Wait a moment for processing
echo -e "${BLUE}\nWaiting a moment...${NC}"
sleep 1

# Get list of objects for the image
echo -e "${BLUE}\nFetching object list...${NC}"
OBJECT_LIST_RESPONSE=$(curl -s "${API_URL}/images/${IMAGE_ID}/objects")

echo -e "${YELLOW}Object list response:${NC}"
echo "$OBJECT_LIST_RESPONSE" | jq '.'

# Check if objects were detected
OBJECT_COUNT=$(echo "$OBJECT_LIST_RESPONSE" | jq '.count // 0')
if [ "$OBJECT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Found $OBJECT_COUNT objects!${NC}"
    echo -e "${YELLOW}\nObject summaries:${NC}"
    echo "$OBJECT_LIST_RESPONSE" | jq '.objects[]'
    
    # Get details of first object
    FIRST_OBJECT_ID=$(echo "$OBJECT_LIST_RESPONSE" | jq -r '.objects[0].object_id // empty')
    if [ ! -z "$FIRST_OBJECT_ID" ]; then
        echo -e "${BLUE}\nFetching details for first object: $FIRST_OBJECT_ID${NC}"
        OBJECT_DETAIL=$(curl -s "${API_URL}/images/${IMAGE_ID}/objects/${FIRST_OBJECT_ID}")
        echo -e "${YELLOW}Object detail:${NC}"
        echo "$OBJECT_DETAIL" | jq '.'
    fi
else
    echo -e "${YELLOW}⚠️  No objects detected${NC}"
fi

# Summary
echo -e "${BLUE}\n===================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}===================================${NC}"
echo -e "Image ID: ${IMAGE_ID}"
echo -e "Objects detected: ${OBJECT_COUNT}"
echo -e "Backend mode: ${MODE}"