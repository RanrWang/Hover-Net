import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def union_image_mask(image_folder,mask_folder,save_folder):
    # 读取原图
    for mask_name in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder,mask_name)
        im_name = mask_name.replace('.npy','.png')
        image_path = os.path.join(image_folder,im_name)
        save_path = os.path.join(save_folder,im_name)
        image = cv2.imread(image_path)
        # image = cv2.resize(image,(10240,10240))
        # 读取分割mask，这里本数据集中是白色背景黑色mask
        mask = np.load(mask_path)
        for i in range(mask.shape[0]):
            mmask = mask[i,:,:]
            mmask = mmask.astype(np.uint8)
            contours, _ = cv2.findContours(mmask,mode=cv2.RETR_LIST,method=cv2.CHAIN_APPROX_NONE)
            if i ==0:
                image = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)
            if i ==1:
                image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            if i ==2:
                image = cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
            if i ==3:
                image = cv2.drawContours(image, contours, -1, (0, 0, 0), 2)

        # # 打开画了轮廓之后的图像
        # plt.imshow(image)
        # plt.show()
        # 保存图像
        cv2.imwrite('{}'.format(save_path), image)
image_folder='/home/ranran/desktop/hover-multi-class（copy）/hover_original_model_pannuke/ConSep_dataset/images'
mask_folder = './consep/multi-mask/'
save_folder = './overlap/'
union_image_mask(image_folder,mask_folder,save_folder)