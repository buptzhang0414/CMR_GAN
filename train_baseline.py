import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
import torchvision.utils as vutils
import torch as th
from util.metrics import PSNR, SSIM
import matplotlib
import matplotlib.image as mpimg
from PIL import Image
#from util.utils_mkdir import prepare_dirs_and_logger, save_config

def train(opt, data_loader, model, visualizer):
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    total_steps = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            # save valid

            if opt.epoch_count == epoch:
                valid_cat = th.cat((data['A'], data['B']), 0)
                vutils.save_image(
                    valid_cat, '{}/deblur{}.jpg'.format(save_dir, total_steps), nrow=opt.batchSize)

            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                results = model.get_current_visuals()
               # psnrMetric = PSNR(
                #    results['Restored_Train'], results['Sharp_Train'])
               # print('PSNR on Train = %f' %
                #      (psnrMetric))
                #visualizer.display_current_results(results, epoch)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                with open("./errors.txt","a+") as f:
                    f.write(str(errors)+"\n")
                results = model.get_current_visuals()
                if(os.path.exists(str(save_dir)+'/blur/')):
                    print('exist')
                else:
                    os.makedirs(str(save_dir)+'/blur/')
                if(os.path.exists(str(save_dir)+'/sharp/')):
                    print('exist')
                else:
                    os.makedirs(str(save_dir)+'/sharp/')
                if(os.path.exists(str(save_dir)+'/restored/')):
                    print('exist')
                else:
                    os.makedirs(str(save_dir)+'/restored/')

                import cv2
                results['Blurred_Train'].save('{}/blur/blur{}.jpg'.format(save_dir, total_steps))
                results['Sharp_Train'].save('{}/sharp/sharp{}.jpg'.format(save_dir, total_steps))
                results['Restored_Train'].save('{}/restored/restored{}.jpg'.format(save_dir, total_steps))
                # cv2.imwrite('{}/blur/blur{}.jpg'.format(save_dir, total_steps),results['Blurred_Train'].reshape((256,256,3)))
                # cv2.imwrite('{}/sharp/sharp{}.jpg'.format(save_dir, total_steps),results['Sharp_Train'].reshape((256, 256, 3)))
                # cv2.imwrite('{}/restored/restored{}.jpg'.format(save_dir, total_steps),results['Restored_Train'].reshape((256, 256, 3)))
#                matplotlib.image.imsave('{}/blur/blur{}_1.jpg'.format(save_dir, total_steps), results['Blurred_Train'])
#                matplotlib.image.imsave('{}/sharp/sharp{}.jpg'.format(save_dir, total_steps), results['Sharp_Train'])
#                matplotlib.image.imsave('{}/restored/restored{}.jpg'.format(save_dir, total_steps), results['Restored_Train'])
               # result = th.cat((th.FloatTensor(results['Blurred_Train']/255),th.FloatTensor(results['Sharp_Train']/255)),0)
               # torch_result = th.FloatTensor(result)
               # print(type(torch_result))
               # print(type(result),"jjjjjjjjjjjjjjjjj")
               # vutils.save_image(result, '{}/deblur{}.jpg'.format(save_dir, total_steps), nrow=opt.batchSize)
               # t = (time.time() - iter_start_time) / opt.batchSize
               # visualizer.print_current_errors(epoch, epoch_iter, errors, t)
               # if opt.display_id > 0:
                #    visualizer.plot_current_errors(epoch, float(
                 #       epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if epoch > opt.niter:
            model.update_learning_rate()


opt = TrainOptions().parse()
print("ss")
print(opt)
print("sss")
# prepare_dirs_and_logger(opt)
# save_config(opt)

data_loader = CreateDataLoader(opt)
model = create_model(opt)
visualizer = Visualizer(opt)
train(opt, data_loader, model, visualizer)
