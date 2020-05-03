from __future__ import  absolute_import
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
		for thresh in np.linspace(0.2, 0.95, 7):
			model.eval()
			pred_bboxes, pred_labels, pred_scores = list(), list(), list()
			ori_imgs, gt_bboxes, gt_labels, gt_difficults = list(), list(), list(), list()
			for ii, (ori_img, imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
				#print("img: ", len(imgs), "\n", imgs, "\n", "boxes: ", gt_bboxes, "\n", "label: ", gt_labels_)
				# print('index: ', ii, "gt_bboxes_ shape: ", len(gt_bboxes_), '\n', "gt_bboxes_", gt_bboxes_, '\n', '*' * 80, '\n')
				for gt_bbox in gt_bboxes_.numpy():
					for gt in gt_bbox:
						gt[0], gt[1], gt[2], gt[3] = gt[1], gt[0], gt[3], gt[2]
				sizes = [sizes[0][0].item(), sizes[1][0].item()]
				pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
				# print('index: ', ii, "pred_bboxes_ shape: ", len(pred_bboxes_), '\n', "pred_bboxes_", pred_bboxes_, '\n', '*' * 80, '\n')
				for pred_bbox in pred_bboxes_:
					for pred in pred_bbox:
						pred[0], pred[1], pred[2], pred[3] = pred[1], pred[0], pred[3], pred[2]
				# print("pred_bboxes_shape: ", len(pred_bboxes_), "\n", "pred_bboxes: ", pred_bboxes_, '\n')
				gt_bboxes += list(gt_bboxes_.numpy())
				gt_labels += list(gt_labels_.numpy())
				gt_difficults += list(gt_difficults_.numpy())
				pred_bboxes += pred_bboxes_
				pred_labels += pred_labels_
				pred_scores += pred_scores_
				print('index:', ii, 'ori_img shape: ', ori_img.shape, '\n', 'ori_img: ', ori_img, '\n', '*' * 80)
				ori_img = ori_img.squeeze().numpy().transpose(1, 2, 0)
				print('index:', ii, 'ori_img shape after squezz and transpose: ', ori_img.shape, '\n', 'ori_img: ', ori_img, '\n', '*' * 80)
				ori_imgs.append(ori_img)


				if ii == test_num: break
			# 612 bboxs in total
			# print("ALL gt_bboxes shape: ", len(gt_bboxes), '\n', "ALL pred_bboxes_", gt_bboxes, '\n',
			# 	  '*' * 80, '\n')
			# print("ALL pred_bboxes shape: ", len(pred_bboxes), '\n', "ALL pred_bboxes_", pred_bboxes, '\n',
			# 	  '*' * 80, '\n')
			#print("pred bboxes: ", pred_bboxes, "\n", "pred labels: ", pred_labels, "\n", "pred scores: ", pred_scores)
			eval = Metric(visualize=True, mode='center', visualization_root='/data1/zinan_xiong/{:0.3f}'.format(thresh))
			for i in range(len(pred_bboxes)):
				pred_bbox = pred_bboxes[i]
				target_bbox = gt_bboxes[i]
				pred_score = pred_scores[i]
				pred_list = []
				target_list = target_bbox
				# print('index: ', i, 'pred_bbox', pred_bbox, '\n', '*' * 80)
				# print('index: ', i, 'target_list', target_list, '\n', '*' * 80)
				# print('index: ', i, 'pred_score', pred_score, '\n', '*' * 80)
				combination_bbox_score = list(zip(pred_bbox, pred_score))
				# print('index: ', i, 'combination_bbox_score', combination_bbox_score, '\n', '*' * 80)
				for combination in combination_bbox_score:
					if combination[1] > thresh:
						pred_list.append(combination[0])
						# target_list.append(j[1])
				# print('index: ', i, 'pred_list', pred_list, '\n', '*' * 80)
				image = cv2.cvtColor(ori_imgs[i], cv2.COLOR_BGR2RGB)
				eval.eval_add_result(target_list, pred_list, image=image, image_name=i)
			precision, recall, pred_bbox_count = eval.get_result()
			F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
			F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
			print("detect time: ", time.time() - st)
			print("Threshold: {:5f}, Prec: {:5f}, Rec: {:5f}, F1: {:5f}, F2: {:5f}, pred_bbox_count: {}".format(thresh, precision, recall, F1, F2, pred_bbox_count))
			saved_results = {"gt_bboxes": gt_bboxes, "gt_labels": gt_labels, "gt_difficults": gt_difficults, "pred_bboxes": pred_bboxes, "pred_labels": pred_labels, "pred_scores": pred_scores, "filtered_pred_bbox": pred_list, "filtered_target_bbox": target_list}
			file_name = 'threshold=' + str(thresh)
			outfile = open(file_name, 'wb')
			pickle.dump(saved_results, outfile)
			outfile.close()
			# print("pred_bboxes size", len(pred_bboxes), pred_bboxes, "\n", "pred_labels size", len(pred_labels), pred_labels, "\n", "pred_scores size", len(pred_scores), pred_scores)
			# print("precision length", len(result[0][1]), "\n",  "precision: ", result[0][1], "\n", "recall legnth: ", len(result[1][1]), "\n", "recall: ", result[1][1])



def main():
    model_path = '/data1/zinan_xiong/simple-faster-rcnn-pytorch/checkpoints/saved/fasterrcnn_01270054'
    #model_path = glob.glob(os.path.join(base_dir, '*.pth'))
    #print(model_path)
    
    if model_path:
        faster_rcnn = FasterRCNNVGG16().cuda()
        print('model construct completed')
        
        testset = TestDataset('/data2/dechunwang/dataset/', split='test')
        test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
									   pin_memory=True
                                       )
        #model = trainer.load_state_dict(torch.load(model_path)['model'])
        state_dict = torch.load(model_path)
        faster_rcnn.load_state_dict(state_dict['model'])
        evaluation_result = eval(test_dataloader, faster_rcnn, len(testset))









if __name__ == '__main__':
    st = time.time()
    main()
