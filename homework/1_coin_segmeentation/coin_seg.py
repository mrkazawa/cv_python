import numpy as np
import cv2

DIR = 'img'
IMAGES = ('lvl1.jpg', 'lvl2.jpg', 'lvl3.jpg', 'lvl4.jpg')
WIDTH = 500

def run_main():
    # load the image
    img = cv2.imread(DIR+'/'+IMAGES[0])
    # resize the image
    img_scale = WIDTH / img.shape[1]  # width
    new_x, new_y = img.shape[1] * img_scale, img.shape[0] * img_scale
    img = cv2.resize(img, (int(new_x), int(new_y)))
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blurring
    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=4)
    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if len(cnt) < 5:
            continue
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 4000:
            continue
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(img, ellipse, (0, 255, 0), 2)

    cv2.imshow("Morphological Closing", closing)
    cv2.imshow("Adaptive Thresholding", thresh)
    cv2.imshow('Contours', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_main()
