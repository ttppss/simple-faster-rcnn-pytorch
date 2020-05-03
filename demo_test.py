import cv2

img_path = '/data2/dechunwang/dataset/large_dataset/Original/49.png'
pic = cv2.imread(img_path)
new_pic = cv2.imwrite('new_49.jpg', pic)