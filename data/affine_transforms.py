"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
zyx add
"""

import math
import random
import torch as th
import cv2

from data.affine_util import th_affine2d, th_perspect2d, normalize_image, exchange_landmarks
import numpy as np


class AffineCompose(object):

    def __init__(self,
                 rotation_range,
                 translation_range,
                 zoom_range,
                 output_img_width,
                 output_img_height,
                 mirror=False,
                 corr_list=None,
                 normalise=False,
                 normalisation_type='regular',
                 ):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.output_img_width = output_img_width
        self.output_img_height = output_img_height
        self.mirror = mirror
        self.corr_list = corr_list
        self.normalise = normalise
        self.normalisation_type = normalisation_type

    def __call__(self, *inputs):
        input_img_width = inputs[0].shape[1]
        input_img_height = inputs[0].shape[2]
        print(inputs[0].shape,"EEE")
        #input_img_width = self.fine_size
        #input_img_height = self.fine_size
        rotate = random.uniform(-self.rotation_range, self.rotation_range)
        trans_x = random.uniform(-self.translation_range,
                                 self.translation_range)
        trans_y = random.uniform(-self.translation_range,
                                 self.translation_range)
        if not isinstance(self.zoom_range, list) and not isinstance(self.zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # rotate
        transform_matrix = th.FloatTensor(3, 3).zero_()
        center = (input_img_width/2.-0.5, input_img_height/2-0.5)
        M = cv2.getRotationMatrix2D(center, rotate, 1)
        transform_matrix[:2, :] = th.from_numpy(M).float()
        transform_matrix[2, :] = th.FloatTensor([[0, 0, 1]])
        # translate
        transform_matrix[0, 2] += trans_x
        transform_matrix[1, 2] += trans_y
        # zoom
        for i in range(3):
            transform_matrix[0, i] *= zoom
            transform_matrix[1, i] *= zoom
        transform_matrix[0, 2] += (1.0 - zoom) * center[0]
        transform_matrix[1, 2] += (1.0 - zoom) * center[1]
        # if needed, apply crop together with affine to accelerate
        transform_matrix[0, 2] -= (input_img_width-self.output_img_width) / 2.0
        transform_matrix[1, 2] -= (input_img_height -
                                   self.output_img_height) / 2.0

        # mirror about x axis in cropped image
        do_mirror = False
        if self.mirror:
            mirror_rng = random.uniform(0., 1.)
            if mirror_rng > 0.5:
                do_mirror = True
        if do_mirror:
            transform_matrix[0, 0] = -transform_matrix[0, 0]
            transform_matrix[0, 1] = -transform_matrix[0, 1]
            transform_matrix[0, 2] = float(
                self.output_img_width)-transform_matrix[0, 2]

        outputs = []
        for idx, _input in enumerate(inputs):
            print(len(_input.shape),"oiio")
            print(_input.shape)
            # input: heatmap_64, face_256
            if len(_input.shape) == 3:
                is_landmarks = False
            else:
                is_landmarks = True
            #input_tf = th_affine2d(_input,
             #                      transform_matrix,
              #                     output_img_width=self.output_img_width,
               #                    output_img_height=self.output_img_height,
                #                   is_landmarks=is_landmarks)
            input_tf=_input
            if is_landmarks and do_mirror and isinstance(self.corr_list, np.ndarray):
                # input_tf.shape: (1L, 68L, 2L)
                # print("mirror!")
                input_tf = exchange_landmarks(input_tf, self.corr_list)
            if (not is_landmarks) and self.normalise:
                # input_tf.shape: (1L/3L, 256L, 256L)
                print(_input.shape,"000998")
                print(input_tf.shape,"98765")
                if _input.shape[0] == 3:
                    input_tf = normalize_image(
                        input_tf, self.normalisation_type)
                else:
                    # for heatmap ground truth generation
                    input_tf = normalize_image(input_tf, 'regular')

            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]


class PerspectCompose(object):

    def __init__(self,
                 rotation_range,
                 translation_range,
                 zoom_range,
                 output_img_width,
                 output_img_height,
                 fine_size,
                 mirror=False,
                 corr_list=None,
                 normalise=False,
                 normalisation_type='regular',
                 ):

        self.fine_size = fine_size
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.output_img_width = output_img_width
        self.output_img_height = output_img_height
        self.mirror = mirror
        self.corr_list = corr_list
        self.normalise = normalise
        self.normalisation_type = normalisation_type

    def __call__(self, *inputs):
        #input_img_width = inputs[0].size(1)
        #input_img_height = inputs[0].size(2)
        input_img_width = self.fine_size
        input_img_height = self.fine_size
        '''
        rotate = random.uniform(-self.rotation_range, self.rotation_range)
        trans_x = random.uniform(-self.translation_range, self.translation_range)
        trans_y = random.uniform(-self.translation_range, self.translation_range)
        if not isinstance(self.zoom_range, list) and not isinstance(self.zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # rotate
        transform_matrix = th.FloatTensor(3, 3).zero_()
        center = (input_img_width/2.-0.5, input_img_height/2-0.5)
        M = cv2.getRotationMatrix2D(center, rotate, 1)
        transform_matrix[:2,:] = th.from_numpy(M).float()
        transform_matrix[2,:] = th.FloatTensor([[0, 0, 1]])
        # translate
        transform_matrix[0,2] += trans_x
        transform_matrix[1,2] += trans_y
        # zoom
        for i in range(3):
            transform_matrix[0,i] *= zoom
            transform_matrix[1,i] *= zoom
        transform_matrix[0,2] += (1.0 - zoom) * center[0]
        transform_matrix[1,2] += (1.0 - zoom) * center[1]
        # if needed, apply crop together with affine to accelerate
        transform_matrix[0,2] -= (input_img_width-self.output_img_width) / 2.0;
        transform_matrix[1,2] -= (input_img_height-self.output_img_height) / 2.0;
        '''
        err = 40
        deta = input_img_height - err
        pt = np.random.randint(0, err, [4, 2])
        pts1 = np.float32([[0, 0], [input_img_height, 0], [0, input_img_height], [
                          input_img_height, input_img_height]])
        pts2 = np.float32([pt[0], pt[1] + np.array([deta, 0]), pt[2] +
                           np.array([0, deta]), pt[3] + np.array([deta, deta])])
        #pts2 = pts1
        transform_matrix = th.FloatTensor(
            cv2.getPerspectiveTransform(pts1, pts2))
        # mirror about x axis in cropped image
        do_mirror = False
        if self.mirror:
            mirror_rng = random.uniform(0., 1.)
            if mirror_rng > 0.5:
                do_mirror = True
        if do_mirror:
            transform_matrix[0, 0] = -transform_matrix[0, 0]
            transform_matrix[0, 1] = -transform_matrix[0, 1]
            transform_matrix[0, 2] = float(
                self.output_img_width)-transform_matrix[0, 2]

        #transform_matrix[2, :] = transform_matrix[2, :] + th.FloatTensor(1, 3)*1

        outputs = []
        for idx, _input in enumerate(inputs):
            # input: heatmap_64, face_256
            if _input.dim() == 3:
                is_landmarks = False
            else:
                is_landmarks = True
            input_tf = th_perspect2d(_input,
                                     transform_matrix,
                                     output_img_width=self.output_img_width,
                                     output_img_height=self.output_img_height,
                                     is_landmarks=is_landmarks)
            if is_landmarks and do_mirror and isinstance(self.corr_list, np.ndarray):
                # input_tf.shape: (1L, 68L, 2L)
                # print("mirror!")
                input_tf = exchange_landmarks(input_tf, self.corr_list)
            if (not is_landmarks) and self.normalise:
                # input_tf.shape: (1L/3L, 256L, 256L)
                if _input.shape[0] == 3:
                    print(self.normalisation_type)
                    input_tf = normalize_image(
                        input_tf, self.normalisation_type)
                else:
                    print("rrrr")
                    # for heatmap ground truth generation
                    input_tf = normalize_image(input_tf, 'regular')

            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]
