import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime

# If Tesseract is not in your PATH, uncomment and set the path accordingly
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess the image to enhance OCR accuracy.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Edge detection
    edged = cv2.Canny(gray, 30, 200)
    return edged, gray

def find_label_contour(edged, image):
    """
    Find the contour that likely represents the label.
    """
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours based on area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    label_contour = None

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Assume the label is a quadrilateral
        if len(approx) == 4:
            label_contour = approx
            break

    if label_contour is None:
        return None

    # Create a mask and extract the label
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [label_contour], -1, (255, 255, 255), -1)
    extracted = cv2.bitwise_and(image, mask)
    
    # Crop the extracted label
    x, y, w, h = cv2.boundingRect(label_contour)
    cropped = image[y:y+h, x:x+w]
    return cropped

def ocr_core(image):
    """
    Perform OCR on the processed image.
    """
    # Convert image to RGB (pytesseract expects RGB)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use Tesseract to do OCR on the image
    custom_config = r'--oem 3 --psm 6'  # OEM 3: Default, PSM 6: Assume a single uniform block of text
    text = pytesseract.image_to_string(rgb, config=custom_config)
    return text

def parse_details(text):
    """
    Parse the OCR text to extract quality, expiry date, and manufacturing date.
    """
    quality = None
    expiry_date = None
    manufacturing_date = None

    # Example patterns (adjust based on label format)
    quality_pattern = re.compile(r'Quality\s*[:\-]\s*(\w+)', re.IGNORECASE)
    expiry_pattern = re.compile(r'Expiry\s*Date\s*[:\-]\s*([\d/.-]+)', re.IGNORECASE)
    manufacturing_pattern = re.compile(r'Manufacturing\s*Date\s*[:\-]\s*([\d/.-]+)', re.IGNORECASE)

    quality_match = quality_pattern.search(text)
    expiry_match = expiry_pattern.search(text)
    manufacturing_match = manufacturing_pattern.search(text)

    if quality_match:
        quality = quality_match.group(1).strip()
    if expiry_match:
        expiry_date = expiry_match.group(1).strip()
    if manufacturing_match:
        manufacturing_date = manufacturing_match.group(1).strip()

    # Optionally, convert dates to standard format
    for date_str in [expiry_date, manufacturing_date]:
        if date_str:
            try:
                parsed_date = datetime.strptime(date_str, '%d/%m/%Y')  # Adjust format as needed
                formatted_date = parsed_date.strftime('%Y-%m-%d')
                if 'expiry_date' in locals() and date_str == expiry_date:
                    expiry_date = formatted_date
                if 'manufacturing_date' in locals() and date_str == manufacturing_date:
                    manufacturing_date = formatted_date
            except ValueError:
                pass  # Handle invalid date formats as needed

    return {
        'quality': quality,
        'expiry_date': expiry_date,
        'manufacturing_date': manufacturing_date
    }

def main():
    # Initialize video capture (0 for default camera or provide video file path)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Preprocess the frame
        edged, gray = preprocess_image(frame)

        # Find and extract label
        label = find_label_contour(edged, frame)
        if label is not None:
            # Perform OCR on the label
            text = ocr_core(label)
            print("OCR Text:")
            print(text)

            # Parse the details
            details = parse_details(text)
            print("Extracted Details:")
            print(details)

            # Draw contour and display
            cv2.drawContours(frame, [label_contour := cv2.boundingRect(find_label_contour(edged, frame))], -1, (0, 255, 0), 2)
            cv2.putText(frame, f"Quality: {details['quality']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Expiry Date: {details['expiry_date']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Manufacturing Date: {details['manufacturing_date']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Conveyor Belt OCR', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
