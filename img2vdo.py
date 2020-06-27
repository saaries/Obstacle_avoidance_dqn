import cv2
import os

im_dir = '/Users/saaries/Downloads/11git/img'

video_dir = '/Users/saaries/Downloads/11git/saveVideo.avi'

fps = 24

num = 438

img_size = (160, 160)

# opencv3.0
fourcc = cv2.VideoWriter_fourcc('m','p','4','2')

videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)

for i in range(0,num):
    im_name = os.path.join(im_dir, str(i)+'.bmp')
    print(im_name)
    
    frame = cv2.imread(im_name)

    videoWriter.write(frame)
    cv2.imshow('rr', frame)
    cv2.waitKey(30)
    videoWriter.write(frame)

videoWriter.release()
print('finish')
