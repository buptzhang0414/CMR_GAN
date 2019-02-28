from .conditional_gan_model import ConditionalGAN
from .baseline_model import BaselineModel


def create_model(opt):
    model = None
    if opt.model == 'test':
        print(opt.dataset_mode)
      #  assert (opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'baseline':
        model = BaselineModel()
    else:
        model = ConditionalGAN()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
