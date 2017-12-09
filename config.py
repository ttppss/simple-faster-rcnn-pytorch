
from pprint import pprint
class Config:
    # data
    voc_data_dir = '/mnt/3/VOC/VOCdevkit/VOC2007/'
    min_size = 600
    max_size = 1000
    num_workers=8

    # sigma for l1_smooth_loss
    rpn_sigma = 3. 
    roi_sigma = 1.

    # param for optimizer
    weight_decay = 0.0005
    # lr = 1e-3
    lr1 = 1e-4 # extractor
    lr2 = 1e-3 # rpn
    lr3 = 1e-3 # roi head

    # visualization
    env = 'faster-rcnn' #visdom env
    port = 8097 
    plot_every = 40 # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    stage = 1 # 4 stage training, -1 for train end2end
    epoch = 100


    # mean and std
    loc_normalize_mean=(0., 0., 0., 0.)
    loc_normalize_std=(0.1, 0.1, 0.2, 0.2)

    # debug
    debug_file = '/tmp/debugf'

    # model
    load_path = '/mnt/3/rpn.pth'

    def _parse(self,kwargs):
        state_dict = self._state_dict()
        for k,v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' %k)
            setattr(self,k,v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k:getattr(self,k) for k,_ in Config.__dict__.items() \
                                        if not k.startswith('_')}

opt = Config()