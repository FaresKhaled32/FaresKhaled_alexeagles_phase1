import cv2 as cv
import numpy as np

def thesholdingtheimg(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY_INV)
    return thresh

def find_defects(idealimg, sampleimg):
    idealthresh = thesholdingtheimg(idealimg)
    samplethresh = thesholdingtheimg(sampleimg)
    bitwisediff = cv.bitwise_xor(idealthresh, samplethresh)
    contours, _ = cv.findContours(bitwisediff, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    broken_count = 0
    worn_count = 0

    for cnt in contours: #i used chatgpt in this part to get the worn and broken teeth
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if area > 300:  
            broken_count += 1
        elif aspect_ratio > 5.0 or area > 100:  
            worn_count += 1

    return broken_count, worn_count, bitwisediff

def main():
    idealimage = cv.imread('ideal.jpg')
    sampleimages = [cv.imread(f'sample{i}.jpg') for i in range(2, 6)]

    for i, sampleimage in enumerate(sampleimages):
        broken_count, worn_count, bitwisediff = find_defects(idealimage, sampleimage)
        print(f"Sample {i+1}: {broken_count} broken teeth, {worn_count} worn teeth")
        cv.imshow(f'Sample {i+1} Differences', bitwisediff)
        cv.waitKey(0)

if __name__ == "__main__":
    main()
