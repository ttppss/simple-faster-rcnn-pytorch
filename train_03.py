from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
import torch
from tqdm import tqdm
from data.polyp_dataset import Polypcoco_anchorfree
from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from metric_polyp import Metric


# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num, thresh = 0.01):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break
    #print("gt_bboxes: ", len(gt_bboxes), gt_bboxes)
    #print("gt_labels: ", len(gt_labels), gt_labels)
    #print("gt_difficults: ", len(gt_difficults), gt_difficults)

    #print("pred_bboxes: ", len(pred_bboxes), pred_bboxes)
    #print("pred_labels: ", len(pred_labels), pred_labels)
    #print("pred_scores: ", len(pred_scores), pred_scores)

    #eval = Metric(visualize=False, visualization_root=None)
    #filtered_score = []
    #filtered_pred = []
    #filtered_target = []
    #for i in range(len(pred_bboxes)):
    #    if pred_scores[i] > thresh:
    #        filtered_score.append(pred_scores[i])
    #        filtered_pred.append(pred_bboxes[i])
    #        filtered_target.append(gt_bboxes[i])
        #pred_list = pred_bboxes[i]
        #target_list = gt_bboxes[i]
        #pred_list = [p for p in pred_list if p[4] >= thresh]
        #filtered_p = [p[:4] for p in pred_list]
        #filtered_target = [p for p in target_list]
    #    image = None
    #    eval.eval_add_result(filtered_target, filtered_pred, image=image, image_name=i)

    #precision, recall = evl.get_result()
    #F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
    #F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
    #print("detect time: ", time.time() - st)
    #print("Prec: ", precision, "Rec: ", recall, "F1: ", F1, "F2: ", F2)

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    #print("detect time: ", time.time() - st)
    print("precision and recall: ", result)
    return result



def train(**kwargs):
    opt._parse(kwargs)

    #dataset = Polypcoco_anchorfree('/data2/dechunwang/dataset', split='train')
    #print("dataset length: ", len(dataset))
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    #print(dataloader)
    # for i, sample_image in enumerate(dataloader):
    #     print("data loader output: ", sample_image)

    testset = TestDataset(opt)
    #testset = Polypcoco_anchorfree('/data2/dechunwang/dataset', split='test')
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    #print("test dataloader", test_dataloader)
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    #trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in enumerate(dataloader):
            #print("loader:", img.shape, bbox_.shape, label_.shape, scale.shape)
            scale = at.scalar(scale)
            # img = torch.FloatTensor(img).unsqueeze(0)
            # bbox_ = torch.FloatTensor(bbox_)
            # print("bbox_ shape: ", bbox_.shape)
            # label_ = torch.FloatTensor(label_)
            # print("*" * 100)
            # print("bbox before tocuda: ", bbox_, bbox_.shape)
            # print("*" * 100)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            # print("*" * 100)
            # print("bbox before trainer.step: ", bbox, bbox.shape)
            # print("*" * 100)
            #print(img.shape)
            trainer.train_step(img, bbox, label, scale)


            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                #trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                # gt_img = visdom_bbox(ori_img_,
                #                      at.tonumpy(bbox_[0]),
                #                      at.tonumpy(label_[0]))
                #trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                # pred_img = visdom_bbox(ori_img_,
                #                        at.tonumpy(_bboxes[0]),
                #                        at.tonumpy(_labels[0]).reshape(-1),
                #                        at.tonumpy(_scores[0]))
                #trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                #trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                #trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=len(testset))
        print("result: ", eval_result)
        #trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        #trainer.vis.log(log_info)
        print("log info: ", log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
            print("best: ", best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break
    #torch.save(trainer.state_dict(), '/data1/zinan/simple-faster-rcnn-pytorch/checkpoints/saved/model.pth')

if __name__ == '__main__':
    import fire

    fire.Fire()
