import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
#from util.metrics import PSNR
#from ssim import SSIM
from util.metrics import PSNR,SSIM
from PIL import Image
from skimage.measure import compare_ssim
import numpy as np
import time
import cv2

def getB(array):
    B = 0
    array_pad = np.pad(array, ((1, 1), (1, 1)), 'constant')
    row = array.shape[0]
    col = array.shape[1]
    for i in range(0,row):
        for j in range(0,col):
            if( array_pad[i][j+1]==1 or  array_pad[i+1][j]==1 or array_pad[i+1][j+2]==1 or array_pad[i+2][j+1]==1):
                B = B + 1
    # print(B,"BBBBBBBBBBBB")
    return B

def getC(array):
    C = 0
    array_pad = np.pad(array, ((1, 1), (1, 1)), 'constant')
    row = array.shape[0]
    col = array.shape[1]
    for i in range(0,row):
        for j in range(0,col):
            if(array_pad[i][j]==1 or array_pad[i][j+1]==1 or array_pad[i][j+2]==1 or array_pad[i+1][j]==1 or array_pad[i+1][j+2]==1
                    or array_pad[i+2][j]==1 or array_pad[i+2][j+1]==1 or array_pad[i+2][j+2]==1):
                C = C + 1
    # print(C,"CCCCCCCCCCC")
    return C

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' %
                       (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (
    opt.name, opt.phase, opt.which_epoch))
# test
start =time.clock()
avgPSNR = 0.0
avgSSIM = 0.0
counter = 0
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    counter = i
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals_test()
#    B += float(getB(visuals['fake_B'].reshape((256,256))))
 #   C += float(getC(visuals['fake_B'].reshape((256,256))))
    avgPSNR += PSNR(visuals['fake_B'],visuals['real_A'])
    #pilFake = Image.fromarray(visuals['fake_B'])
    #pilReal = Image.fromarray(visuals['real_A'])
    #avgSSIM += SSIM(pilFake).cw_ssim_value(pilReal)
    avgSSIM +=compare_ssim(visuals['fake_B'].reshape((256,256)),visuals['real_A'].reshape((256,256)))
    img_path = model.get_image_paths()
    print(type(visuals['fake_B'].reshape((256,256))),"type")
    print(visuals['fake_B'].reshape((256,256)),"type")
    # print('process image... %s' % img_path)
    visualizer.save_images_test(webpage, visuals, img_path)
end = time.clock()
print('Running time: %s Seconds'%(end-start))
avgPSNR /= counter
avgSSIM /= counter
print('PSNR = %f, SSIM = %f' %
				  (avgPSNR, avgSSIM))

webpage.save()
