import numpy as np
import cv2
import os.path
#import random
import torchvision.transforms as transforms
import torch
from PIL import Image
from data.base_dataset import BaseDataset
#from data.image_folder import make_dataset
#from PIL import Image
from data.affine_transforms import AffineCompose
from data.base_dataset import BaseDataset, get_transform

class CMRDataset(BaseDataset):
    
    def initialize(self, opt):
        self.opt = opt
        with open(opt.name_list_blur, 'r') as f:
            self.name_list_blur = f.readlines()
        self.name_list_blur = [ii.strip('\n') for ii in self.name_list_blur]
        with open(opt.name_list_sharp, 'r') as f:
            self.name_list_sharp = f.readlines()
        self.name_list_sharp = [ii.strip('\n') for ii in self.name_list_sharp]

        self.dir_blur = opt.dir_blur
        self.dir_sharp = opt.dir_sharp
        self.fine_size = opt.fineSize
        self.phase = opt.phase
        self.transform_align = AffineCompose(rotation_range=5,
                                            translation_range=10,
                                            zoom_range=[0.97, 1.03],
                                            output_img_width=self.fine_size,
                                            output_img_height=self.fine_size,
                                            mirror=True,
                                            corr_list=None,
                                            normalise=True,
                                            normalisation_type='regular',
                                            )


        self.transform = get_transform(opt)
    def __getitem__(self, index):
        blur_path = '{}/{}'.format(self.dir_blur, self.name_list_blur[index])
        sharp_path = '{}/{}'.format(self.dir_sharp, self.name_list_sharp[index])
        # blur_img = cv2.imread(blur_path, 1)#1读出来就是彩色图，0读出来就是灰度图
        # sharp_img = cv2.imread(sharp_path, 1)
        # blur_img =  cv2.resize(blur_img,(self.fine_size,self.fine_size),interpolation=cv2.INTER_CUBIC )
        # sharp_img =  cv2.resize(sharp_img,(self.fine_size,self.fine_size),interpolation=cv2.INTER_CUBIC )
        #print(blur_img.min(), blur_img.max(), 'ereget\n')
        blur_img =Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        blur_img = blur_img.resize((self.fine_size, self.fine_size), Image.ANTIALIAS)
        sharp_img = sharp_img.resize((self.fine_size, self.fine_size), Image.ANTIALIAS)
        # cv2.imwrite('tmp.jpg', blur_img)
        # blur_img = blur_img.reshape((blur_img.shape[2], blur_img.shape[1], blur_img.shape[0]))
        # sharp_img = sharp_img.reshape((sharp_img.shape[2], sharp_img.shape[1], sharp_img.shape[0]))
        # print (blur_img.shape, "iiu")
        # cv2.imwrite('tmp11.jpg', blur_img.reshape(256,256,3))
#        blur_img = blur_img[np.newaxis, :]
 #       sharp_img = sharp_img[np.newaxis, :]
        if(self.phase=="test"):
            blur_img_tensor = torch.from_numpy(blur_img).float()/255.0
            sharp_img_tensor = torch.from_numpy(sharp_img).float()/255.0
            output=[blur_img_tensor, sharp_img_tensor]
        else:
            # blur_img= self.transform(blur_img)
            # sharp_img= self.transform(sharp_img)
            # output = self.transform_align(*[blur_img, sharp_img])
           # print(output,"output2")
            blur_img = self.transform(blur_img)
            sharp_img = self.transform(sharp_img)
#        output = self.transform_align(*[blur_img, sharp_img])
#         print(output[0].min(), output[0].max(), 'fgghghgh\n')
#         blur_img_tensor = torch.from_numpy(blur_img).float()
#         sharp_img_tensor = torch.from_numpy(sharp_img).float()
        output = [blur_img, sharp_img]
        # print (output[0].shape, "output[0].shape,cmrdataset")
        # cv2.imwrite("oo.jpg",output[0].numpy().reshape((256,256,3)))
        # cv2.imwrite("oo1.jpg", blur_img)
        # print(type(output[0]),"output[0].type")
        return {'A': output[0], 'B': output[1],
        # return {'A': blur_img, 'B': sharp_img,
                'A_paths': blur_path, 'B_paths': sharp_path}

        # def __getitem__(self, index):
        #     blur_path = '{}/{}'.format(self.dir_blur, self.name_list_blur[index])
        #     sharp_path = '{}/{}'.format(self.dir_sharp, self.name_list_sharp[index])
        #     blur_img = cv2.imread(blur_path, 1)  # 1读出来就是彩色图，0读出来就是灰度图
        #     sharp_img = cv2.imread(sharp_path, 1)
        #     blur_img = cv2.resize(blur_img, (self.fine_size, self.fine_size), interpolation=cv2.INTER_CUBIC)
        #     sharp_img = cv2.resize(sharp_img, (self.fine_size, self.fine_size), interpolation=cv2.INTER_CUBIC)
        #     # print(blur_img.min(), blur_img.max(), 'ereget\n')
        #     print(blur_img.shape, "ii")
        #     #
        #     cv2.imwrite('tmp.jpg', blur_img)
        #     blur_img = blur_img.reshape((blur_img.shape[2], blur_img.shape[1], blur_img.shape[0]))
        #     sharp_img = sharp_img.reshape((sharp_img.shape[2], sharp_img.shape[1], sharp_img.shape[0]))
        #     print(blur_img.shape, "iiu")
        #     # cv2.imwrite('tmp11.jpg', blur_img.reshape(256,256,3))
        #     #        blur_img = blur_img[np.newaxis, :]
        #     #       sharp_img = sharp_img[np.newaxis, :]
        #     if (self.phase == "test"):
        #         blur_img_tensor = torch.from_numpy(blur_img).float() / 255.0
        #         sharp_img_tensor = torch.from_numpy(sharp_img).float() / 255.0
        #         output = [blur_img_tensor, sharp_img_tensor]
        #     else:
        #         # blur_img= self.transform(blur_img)
        #         # sharp_img= self.transform(sharp_img)
        #         output = self.transform_align(*[blur_img, sharp_img])
        #     # print(output,"output2")
        #     # blur_img = self.transform(blur_img)
        #     # sharp_img = self.transform(sharp_img)
        #     #        output = self.transform_align(*[blur_img, sharp_img])
        #     #         print(output[0].min(), output[0].max(), 'fgghghgh\n')
        #     blur_img_tensor = torch.from_numpy(blur_img).float()
        #     sharp_img_tensor = torch.from_numpy(sharp_img).float()
        #     output = [blur_img_tensor, sharp_img_tensor]
        #     print(output[0].shape, "output[0].shape,cmrdataset")
        #     cv2.imwrite("oo.jpg", output[0].numpy().reshape((256, 256, 3)))
        #     # cv2.imwrite("oo1.jpg", blur_img)
        #     print(type(output[0]), "output[0].type")
        #     return {'A': output[0], 'B': output[1],
        #             # return {'A': blur_img, 'B': sharp_img,
        #             'A_paths': blur_path, 'B_paths': sharp_path}
    def __len__(self):
        return len(self.name_list_blur)

    def name(self):
        return 'CMRDataset'
