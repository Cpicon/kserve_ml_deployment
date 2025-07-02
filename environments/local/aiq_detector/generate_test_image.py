from PIL import Image, ImageDraw
import os
import base64
import sys
import json

# Only create test image if no argument is provided
if len(sys.argv) == 1:
    # Create a new image with white background
    width, height = 400, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)

    # Draw a circle (coin) in the center
    center_x, center_y = width // 2, height // 2
    radius = 80
    # Draw a filled circle with golden color (like a coin)
    draw.ellipse([center_x - radius, center_y - radius, 
                  center_x + radius, center_y + radius], 
                  fill='gold', outline='darkgoldenrod', width=3)

    # Add some texture to make it look more like a coin
    inner_radius = radius - 20
    draw.ellipse([center_x - inner_radius, center_y - inner_radius, 
                  center_x + inner_radius, center_y + inner_radius], 
                  outline='goldenrod', width=2)

    # Save the image
    os.makedirs('test_data', exist_ok=True)
    image.save('test_data/test_coin.jpg')
    print("Test image created: test_data/test_coin.jpg") 

def image_to_base64(image_path):
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Use provided image or generated test image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Use the image from dataset as suggested
    image_path = "../../../dataset/0ba4fa31-913c-45be-9e59-bc14fe4f324e_jpg.rf.89dddeb3544e94d2c5f1aa763b85823d.jpg"
    if not os.path.exists(image_path):
        # Fallback to generated test image
        image_path = "test_data/test_coin.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    sys.exit(1)

# Create test input in TensorFlow V1 HTTP API format
test_input = {
    "instances": [
        {
            "image": {
                "b64": image_to_base64(image_path)
            }
        }
    ],
    "parameters": {
        "label": "coin"
    }
}

# Save to JSON file
os.makedirs('test_data', exist_ok=True)
with open('test_data/input.json', 'w') as f:
    json.dump(test_input, f, indent=2)

print(f"Test input created from {image_path}")
print("Saved to: test_data/input.json") 