from __future__ import absolute_import
import os, glob
from utils.config import opt
from data import Polypcoco_anchorfree
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from metric_polyp import Metric
import torch
from torchvision.models import vgg16
from utils.eval_tool import calc_detection_voc_prec_rec
from metric_polyp import *
import time
import pickle


def eval(dataloader, model, test_num):
    with torch.no_grad():
        for thresh in np.linspace(0.2, 0.9, 7):
            model.eval()
            pred_bboxes, pred_labels, pred_scores = list(), list(), list()
            gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
            for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
                # print("img: ", len(imgs), "\n", imgs, "\n", "boxes: ", gt_bboxes, "\n", "label: ", gt_labels_)
                # print("gt_labesl shape: ", gt_labels_)
                sizes = [sizes[0][0].item(), sizes[1][0].item()]
                pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
                # print("pred_bboxes: ", pred_bboxes_, "\n", "pred_labels: ", pred_labels_, "\n", "pred_scores: ", pred_scores_)
                gt_bboxes += list(gt_bboxes_.numpy())
                gt_labels += list(gt_labels_.numpy())
                gt_difficults += list(gt_difficults_.numpy())
                pred_bboxes += pred_bboxes_
                pred_labels += pred_labels_
                pred_scores += pred_scores_

                print("img shape: ", imgs.shape)
                print("pred_bbox_: ", pred_bboxes_)
                print("gt_bboxes_: ", gt_bboxes_)

                img = draw_func(imgs, gt_bboxes_, pred_bboxes_)

                cv2.imwrite('/data1/zinan/fasterrcnn_result_image//{}.jpg'.format(ii), img)

                if ii == test_num: break
            # print("pred bboxes: ", pred_bboxes, "\n", "pred labels: ", pred_labels, "\n", "pred scores: ",
            # pred_scores)
            eval = Metric(visualize=False, visualization_root=None)
            for i in range(len(pred_bboxes)):
                pred_bbox = pred_bboxes[i]
                target_bbox = gt_bboxes[i]
                pred_score = pred_scores[i]
                pred_list = []
                target_list = []
                combination_bbox_score = list(zip(pred_bbox, target_bbox, pred_score))
                # print(combination_bbox_score)
                for j in range(len(pred_bbox)):
                    if combination_bbox_score[0][2] > thresh:
                        pred_list.append(combination_bbox_score[0][0])
                        target_list.append(combination_bbox_score[0][1])
                image = None
            # eval.eval_add_result(target_list, pred_list,image=image, image_name=i)
        # precision, recall, pred_bbox_count = eval.get_result()
        # F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
        # F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
        # print("detect time: ", time.time() - st)
        # print("Threshold: {:5f}, Prec: {:5f}, Rec: {:5f}, F1: {:5f}, F2: {:5f}, pred_bbox_count: {}".format(thresh, precision, recall, F1, F2, pred_bbox_count))
        # saved_results = {"gt_bboxes": gt_bboxes, "gt_labels": gt_labels, "gt_difficults": gt_difficults, "pred_bboxes": pred_bboxes, "pred_labels": pred_labels, "pred_scores": pred_scores, "filtered_pred_bbox": pred_list, "filtered_target_bbox": target_list}
        # file_name = 'threshold=' + str(thresh)
        # outfile = open(file_name, 'wb')
        # pickle.dump(saved_results, outfile)
        # outfile.close()
        # print("pred_bboxes size", len(pred_bboxes), pred_bboxes, "\n", "pred_labels size", len(pred_labels), pred_labels, "\n", "pred_scores size", len(pred_scores), pred_scores)
        # print("precision length", len(result[0][1]), "\n",  "precision: ", result[0][1], "\n", "recall legnth: ", len(result[1][1]), "\n", "recall: ", result[1][1])


def draw_func(imgs, gt_bboxes, pred_bboxes_):
    imgs = imgs
    gt_bboxes = gt_bboxes
    pred_bboxes_ = pred_bboxes_
    for pt in gt_bboxes:
        pt1 = (pt[0], pt[1])
        pt2 = (pt[2], pt[3])
    cv2.rectangle(imgs, pt1, pt2, color='red', thickness=2)

    for pred_bbox in pred_bboxes_:
        print("pred_bbox before: ", pred_bbox)
        pred_bbox = pred_bbox[1, 0, 3, 2]
        print("pred_bbox after: ", pred_bbox)
        for pt in pred_bbox:
            pt1 = (pt[0], pt[1])
            pt2 = (pt[2], pt[3])
        cv2.rectangle(imgs, pt1, pt2, color='green', thickness=2)

        return imgs


def main():
    model_path = '/data1/zinan/simple-faster-rcnn-pytorch/checkpoints/saved/fasterrcnn_01270054'
    # model_path = glob.glob(os.path.join(base_dir, '*.pth'))
    # print(model_path)

    if model_path:
        faster_rcnn = FasterRCNNVGG16().cuda()
        print('model construct completed')
        if not os.path.isdir('/data1/zinan/fasterrcnn_result_image/'):
            os.makedirs('/data1/zinan/fasterrcnn_result_image/')
        testset = Polypcoco_anchorfree('/data1/zinan/xiangya_backup', split='test')
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False, \
                                           pin_memory=True
                                           )
        # model = trainer.load_state_dict(torch.load(model_path)['model'])
        state_dict = torch.load(model_path)
        faster_rcnn.load_state_dict(state_dict['model'])
        evaluation_result = eval(test_dataloader, faster_rcnn, len(testset))


if __name__ == '__main__':
    st = time.time()
    main()
