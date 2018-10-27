import cv2



img = cv2.imread('eval_bv.png')


cv2.rectangle(img, (100, 100), (500, 500), (255, 0, 0), 1)
cv2.imwrite('bb.png', img)