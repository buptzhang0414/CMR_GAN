import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
from .losses import EdgeLoss as lossModel

try:
    xrange          # Python2
except NameError:
    xrange = range  # Python 3


class BaselineModel(BaseModel):
    def name(self):
        return 'BaselineModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # load/define networks
        # Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
        use_parallel = not opt.gan_type == 'wgan-gp'
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel, opt.learn_residual)


        use_sigmoid = opt.gan_type == 'gan'
        self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
        if opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.criticUpdates = 5 if opt.gan_type == 'wgan-gp' else 1

            # define loss functions
            self.contentLoss,self.discLoss = init_loss(opt, self.Tensor)#,self.edgeLoss

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        # print(input_A.shape,"input_A")
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        # print(input_A.shape,"input_A_resize")
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)

        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):
        # Second, G(A) = B
        self.loss_G_Content = self.contentLoss.get_loss(
            self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G_Edge = self.edgeLoss.get_loss(
        #    self.fake_B, self.real_B) * self.opt.lambda_B
        self.loss_G_Gan = self.discLoss.get_g_loss(self.netD, self.real_A, self.fake_B) * self.opt.lambda_B
        #print(self.loss_G_Gan,"sdsddssdddd")
        #print(self.loss_G_Content,"dddd")
        #print(self.loss_G_Edge,"dddddd")
        self.loss_G = self.loss_G_Content  + self.loss_G_Gan #+self.loss_G_Edge
        self.loss_G.backward()
        
        #self.loss_G_Content.backward()
        #self.loss_G_Edge.backward()

    def backward_D(self):
        self.loss_D = self.discLoss.get_loss(self.netD, self.real_A, self.fake_B, self.real_B) * self.opt.lambda_B

        self.loss_D.backward()



    def optimize_parameters(self):
        self.forward()
        for iter_d in xrange(self.criticUpdates):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        # return OrderedDict([('G_L1', self.loss_G_Content.data[0]),('G_gan', self.loss_G_Gan.data[0]),('G_edge', self.loss_G_Edge.data[0]),('D_real+fake', self.loss_D.data[0])])
        return OrderedDict([('G_L1', self.loss_G_Content.data[0]),('G_gan', self.loss_G_Gan.data[0]),('D_real+fake', self.loss_D.data[0])])

    def get_current_visuals(self):
        print (self.real_A.data.shape,"self.real_A.data")
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('Blurred_Train', real_A), ('Restored_Train', fake_B), ('Sharp_Train', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
