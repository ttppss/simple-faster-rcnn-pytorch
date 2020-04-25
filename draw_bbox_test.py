from __future__ import absolute_import
import os, glob
from utils.config import opt
from data.voc_dataset import VOCBboxDataset
from data.dataset import Dataset, TestDataset, inverse_normalize  # , TempDataset
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
        thresh = 0.9
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

            img = draw_func(imgs, gt_bboxes_, pred_bboxes_)

            cv2.imwrite('/data0/zinan_xiong/fasterrcnn_result_image/{}.jpg'.format(ii), img)

            if ii == test_num: break


def draw_func(imgs, gt_bboxes, pred_bboxes_):
    imgs = imgs.numpy()
    print('imgs: ', imgs, 'imgs shape: ', imgs.shape)
    gt_bboxes = gt_bboxes
    pred_bboxes_ = pred_bboxes_
    for pt in gt_bboxes:
        # print('pt: ', pt, 'pt shape: ', pt.shape)
        pt1 = (int(pt[0][0].item()), int(pt[0][1].item()))
        # print('pt1: ', pt1)
        pt2 = (int(pt[0][2].item()), int(pt[0][3].item()))
        cv2.rectangle(imgs, pt1, pt2, (255, 0, 0), 2)

    for pred_bbox in pred_bboxes_:
        # print('pred_bbox: ', pred_bbox, 'pred_bbox shape: ', pred_bbox.shape)
        for pb in pred_bbox:
            pt1 = (int(pb[0].item()), int(pb[1].item()))
            pt2 = (int(pb[2].item()), int(pb[3].item()))
            cv2.rectangle(imgs, pt1, pt2, (0, 255, 0), 2)

        return imgs

        # # print("pred bboxes: ", pred_bboxes, "\n", "pred labels: ", pred_labels, "\n", "pred scores: ", pred_scores)
        # eval = Metric(visualize=False, visualization_root=None)
        # for i in range(len(pred_bboxes)):
        #     pred_bbox = pred_bboxes[i]
        #     target_bbox = gt_bboxes[i]
        #     pred_score = pred_scores[i]
        #     pred_list = []
        #     target_list = []
        #     combination_bbox_score = list(zip(pred_bbox, target_bbox, pred_score))
        #     # print(combination_bbox_score)
        #     for j in range(len(pred_bbox)):
        #         if combination_bbox_score[0][2] > thresh:
        #             pred_list.append(combination_bbox_score[0][0])
        #             target_list.append(combination_bbox_score[0][1])
        #
        #     for pt in pred_list:
        #         pt1 = (pt[0], pt[1])
        #         pt2 = (pt[2], pt[3])
        #
        #         cv2.rectangle(image, pt1, pt2, self.color_maps.score_to_color(pt[4].item()), 2)
        #
        #
        #     image = None
        #     eval.eval_add_result(target_list, pred_list, image=image, image_name=i)
        # precision, recall, pred_bbox_count = eval.get_result()
        # F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
        # F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
        # print("detect time: ", time.time() - st)
        # print("Threshold: {:5f}, Prec: {:5f}, Rec: {:5f}, F1: {:5f}, F2: {:5f}, pred_bbox_count: {}".format(thresh,
        #                                                                                                     precision,
        #                                                                                                     recall,
        #                                                                                                     F1, F2,
        #                                                                                                     pred_bbox_count))
        # saved_results = {"gt_bboxes": gt_bboxes, "gt_labels": gt_labels, "gt_difficults": gt_difficults,
        #                  "pred_bboxes": pred_bboxes, "pred_labels": pred_labels, "pred_scores": pred_scores,
        #                  "filtered_pred_bbox": pred_list, "filtered_target_bbox": target_list}
        # file_name = 'threshold=' + str(thresh)
        # outfile = open(file_name, 'wb')
        # pickle.dump(saved_results, outfile)
        # outfile.close()
        # print("pred_bboxes size", len(pred_bboxes), pred_bboxes, "\n", "pred_labels size", len(pred_labels), pred_labels, "\n", "pred_scores size", len(pred_scores), pred_scores)
        # print("precision length", len(result[0][1]), "\n",  "precision: ", result[0][1], "\n", "recall legnth: ", len(result[1][1]), "\n", "recall: ", result[1][1])


def main():
    model_path = '/data1/zinan_xiong/simple-faster-rcnn-pytorch/checkpoints/saved/fasterrcnn_01270054'
    # model_path = glob.glob(os.path.join(base_dir, '*.pth'))
    # print(model_path)
    image_save_path = '/data0/zinan_xiong/fasterrcnn_result_image/'
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    if model_path:
        faster_rcnn = FasterRCNNVGG16().cuda()
        print('model construct completed')

        # TODO: need to change the dir, and the dataloader structure, to get the file from the correct dir.
        testset = TestDataset('/data1/zinan/xiangya_backup', split='test')
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False,
                                           pin_memory=True
                                           )
        # model = trainer.load_state_dict(torch.load(model_path)['model'])
        state_dict = torch.load(model_path)
        faster_rcnn.load_state_dict(state_dict['model'])
        eval(test_dataloader, faster_rcnn, len(testset))


if __name__ == '__main__':
    st = time.time()
    main()
