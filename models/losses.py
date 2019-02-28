import torch
import torch.nn as nn
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
import torchvision.models as models
import util.util as util
from util.image_pool import ImagePool
from torch.autograd import Variable
from models.networks import Edge
###############################################################################
# Functions
###############################################################################

# add 514

def _numpy2cuda(x):
    x = x.astype(float)
    x = x.reshape((1,1,3,3))
    return torch.FloatTensor(x).cuda()

def _init_Edge():
    Edge_ = Edge().cuda()
#    print(Edge_.state_dict()['grad_x.0.weight'])
    edge_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    edge_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    map_ = {'grad_x.0.weight': _numpy2cuda(edge_x),
            'grad_y.0.weight': _numpy2cuda(edge_y)}

    Edge_.load_state_dict(map_)

    return Edge_




class EdgeLoss():
    def initialize(self):
        self.criterion = nn.MSELoss()
        self.Edge_ = _init_Edge()
        self.batch_size = 10
    def get_loss(self,fakeIm,realIm):
        print(fakeIm.shape,"iiiii")
        print(fakeIm[:,0,:,:].shape,"ppppppppppp"
        )
        print(type(fakeIm[:,0,:,:]),"8888")
        fakeIm=fakeIm[:,0,:,:].data.cpu().numpy().reshape((self.batch_size,1,256,256))
        realIm = realIm[:, 0, :, :].data.cpu().numpy().reshape((self.batch_size, 1, 256, 256))
        print(type(fakeIm))

        fakeIm_R = Variable(torch.from_numpy(fakeIm))
        realIm_R = Variable(torch.from_numpy(realIm))
        print(type(fakeIm_R))
        fake_grad_x ,fake_grad_y =  self.Edge_(fakeIm_R)
        real_grad_x ,real_grad_y = self.Edge_(realIm_R)
       # return (fake_grad_x-real_grad_x)**2+(fake_grad_y - real_grad_y)**2
        return self.criterion(fake_grad_x,real_grad_x.detach())+self.criterion(fake_grad_y,real_grad_y.detach())

class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'DiscLoss'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)
        self.fake_AB_pool = ImagePool(opt.pool_size)

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA, fakeB, realB):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fakeB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(realB)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=True, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB):
        return DiscLoss.get_g_loss(self, net, realA, fakeB)

    def get_loss(self, net, realA, fakeB, realB):
        return DiscLoss.get_loss(self, net, realA, fakeB, realB)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self, opt, tensor):
        DiscLossLS.initialize(self, opt, tensor)
        self.LAMBDA = 10

    def get_g_loss(self, net, realA, fakeB):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fakeB)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(
                                      disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                            ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, realA, fakeB, realB):
        self.D_fake = net.forward(fakeB.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(realB)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(
            net, realB.data, fakeB.data)
        return self.loss_D + gradient_penalty






def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    if opt.model == 'content_gan':
        content_loss = PerceptualLoss()
        content_loss.initialize(nn.MSELoss())
    elif opt.model == 'pix2pix':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
    elif opt.model == 'baseline':
        content_loss = ContentLoss()
        content_loss.initialize(nn.L1Loss())
        # edge_loss = EdgeLoss()
        # edge_loss.initialize()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)

    if opt.gan_type == 'wgan-gp':
        disc_loss = DiscLossWGANGP()
    elif opt.gan_type == 'lsgan':
        disc_loss = DiscLossLS()
    elif opt.gan_type == 'gan':
        disc_loss = DiscLoss()
    else:
        raise ValueError("GAN [%s] not recognized." % opt.gan_type)
    disc_loss.initialize(opt, tensor)
    return  content_loss,disc_loss #edge_loss
   # return disc_loss, content_loss
