import cv2
import numpy as np
from matplotlib import pyplot as plt

def classify_aspect_ratio(ratio):
    """Classify based on the aspect ratio."""
    ratio = round(ratio, 1)
    if ratio >= 3:
        return "(Slender)"
    elif 2.1 <= ratio < 3:
        return "(Medium)"
    elif 1.1 <= ratio < 2.1:
        return "(Bold)"
    else:
        return "(Round)"

def preprocess_image(image_path):
    """Load image and apply binary threshold, filter, morphological ops, and edge detection."""
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    
    kernel_avg = np.ones((5, 5), np.float32) / 9
    filtered = cv2.filter2D(binary, -1, kernel_avg)

    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(filtered, kernel_morph, iterations=1)
    dilated = cv2.dilate(eroded, kernel_morph, iterations=1)

    edges = cv2.Canny(dilated, 100, 200)

    return gray, binary, filtered, eroded, dilated, edges

def analyze_grains(eroded_img):
    """Find contours and classify each grain based on aspect ratio."""
    contours, _ = cv2.findContours(eroded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"No. of rice grains = {len(contours)}")
    total_ar = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if w > h else float(h) / w
        print(f"{round(aspect_ratio, 2)} {classify_aspect_ratio(aspect_ratio)}")
        total_ar += aspect_ratio

    avg_ar = total_ar / len(contours)
    print(f"Average Aspect Ratio = {round(avg_ar, 2)} {classify_aspect_ratio(avg_ar)}")

def display_images(images, titles):
    """Display a list of images with titles using matplotlib."""
    rows, cols = 2, 3
    plt.figure(figsize=(12, 8))
    for idx, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    print("Starting...")
    
    image_path = 'rice.png'
    gray, binary, filtered, eroded, dilated, edges = preprocess_image(image_path)

    analyze_grains(eroded)

    display_images(
        [gray, binary, filtered, eroded, dilated, edges],
        ["Original", "Binary", "Filtered", "Eroded", "Dilated", "Edges"]
    )

if __name__ == "__main__":
    main()
