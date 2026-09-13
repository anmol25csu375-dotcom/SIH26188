from paddleocr import PaddleOCR


# Create the OCR engine
ocr = PaddleOCR(
    lang="en"
)

# Replace this with the path to your synthetic/redacted image
image_path = "sample.jpeg"

# Run OCR
result = ocr.predict(image_path)

print("\nOCR Result:\n")

for page in result:
    print(page)