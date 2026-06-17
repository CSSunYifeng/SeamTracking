import cv2
import numpy as np
import json
import os
ROOT = os.getenv('SEAMTRACKING_ROOT')


class inferencer():
    def __init__(self):
        json_value = None
        with open(os.path.join(ROOT, 'data/map.json'),'r') as json_reader:
            json_value = json.load(json_reader)
        self.path_list = [ os.path.join(ROOT,'data',jpath) for jpath in json_value]

    def random_infer(self,im_src):
        i = np.random.randint(len(self.path_list))
        im_infer = cv2.imread(self.path_list[i])
        result = self.add(im_infer,im_src)
        return result

    def add(self,im_infer,im_src):
        if im_infer.shape != im_src.shape:
            h,w = im_src.shape[:2]
            im_infer = cv2.resize(im_infer,(w,h))
        if im_src.shape[2] == 1:
            im_infer = cv2.cvtColor(im_infer,cv2.COLOR_RGB2GRAY)
            im_infer = np.expand_dims(im_infer,axis=-1)
        result_im = np.maximum(im_infer,im_src)
        return result_im


# # AB
# image_a = cv2.imread(src_path)
# image_b = cv2.imread(src2_path)

# # 
# if image_a.shape != image_b.shape:
#     raise ValueError("AB，。")

# # 
# result_image = cv2.max(image_a, image_b)

# # 
# cv2.imwrite('result_image.jpg', result_image)

