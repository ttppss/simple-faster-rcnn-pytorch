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


def eval(dataloader, model, test_num):
	with torch.no_grad():
		model.eval()
		pred_bboxes, pred_labels, pred_scores = list(), list(), list()
		gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
		for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
			print("img: ", len(imgs), "\n", imgs, "\n", "boxes: ", gt_bboxes)
			sizes = [sizes[0][0].item(), sizes[1][0].item()]
			pred_bboxes_, pred_labels_, pred_scores_ = model.predict(imgs, [sizes])
			print("pred_bboxes: ", pred_bboxes_, "\n", "pred_labels: ", pred_labels_, "\n", "pred_scores: ", pred_scores_)
			gt_bboxes += list(gt_bboxes_.numpy())
			gt_labels += list(gt_labels_.numpy())
			gt_difficults += list(gt_difficults_.numpy())
			pred_bboxes += pred_bboxes_
			pred_labels += pred_labels_
			pred_scores += pred_scores_
			if ii == test_num: break
		#print("pred bboxes: ", pred_bboxes, "\n", "pred labels: ", pred_labels, "\n", "pred scores: ", pred_scores)
		result = calc_detection_voc_prec_rec(
			pred_bboxes, pred_labels, pred_scores,
			gt_bboxes, gt_labels, gt_difficults)

		print("precision and recall: ", result)



def main():
    model_path = '/data1/zinan/simple-faster-rcnn-pytorch/checkpoints/saved/fasterrcnn_0126'
    #model_path = glob.glob(os.path.join(base_dir, '*.pth'))
    #print(model_path)
    
    if model_path:
        dataset = Dataset(opt)
        faster_rcnn = FasterRCNNVGG16().cuda()
        print('model construct completed')
        
        testset = TestDataset(opt)
        test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
        #model = trainer.load_state_dict(torch.load(model_path)['model'])
        state_dict = torch.load(model_path)
        faster_rcnn.load_state_dict(state_dict['model'])
        evaluation_result = eval(test_dataloader, faster_rcnn, len(testset))









if __name__ == '__main__':
    main()
