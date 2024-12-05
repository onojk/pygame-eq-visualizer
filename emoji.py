from PIL import Image, ImageDraw, ImageFont

# Configuration
EMOJI_SIZE = 72  # Base size for emojis (width and height)
EMOJI_OUTPUT_SIZE = 144  # Scaled size for better visibility
EMOJI_NAMES = {
    "happy": "😊",
    "distressed": "😟",
    "collision": "😵",
}

# Generate emoji images
def generate_emoji_images():
    emoji_images = {}
    font = ImageFont.truetype("DejaVuSans.ttf", EMOJI_SIZE)  # Adjust font size and path if needed
    for name, emoji in EMOJI_NAMES.items():
        # Create a blank RGBA image
        img = Image.new("RGBA", (EMOJI_OUTPUT_SIZE, EMOJI_OUTPUT_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Calculate text bounding box and position to center the emoji
        text_bbox = draw.textbbox((0, 0), emoji, font=font)  # New method in Pillow
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (EMOJI_OUTPUT_SIZE - text_width) // 2
        text_y = (EMOJI_OUTPUT_SIZE - text_height) // 2

        # Draw the emoji onto the image
        draw.text((text_x, text_y), emoji, font=font, fill="white")

        # Save the generated emoji
        img.save(f"{name}.png")
        emoji_images[name] = f"{name}.png"

    return emoji_images

# Generate and save emoji images
generate_emoji_images()

