import base64
from io import BytesIO
from PIL import Image as PILImage


def decode_from_base64(data: str) -> PILImage.Image:
    """Decode a base64 image payload into a PIL image."""
    image_bytes = base64.b64decode(data)
    return PILImage.open(BytesIO(image_bytes))
