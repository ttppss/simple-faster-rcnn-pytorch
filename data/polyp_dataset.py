import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import json
import os
import glob
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageFile
from skimage import measure
import imageio
import sys
from data.dataset import Transform as FSTrans
from data.dataset import caffe_normalize, pytorch_normalze
from skimage import transform as sktsf
from utils.config import opt


ImageFile.LOAD_TRUNCATED_IMAGES = True


class PolypAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        pass

    def __call__(self, target, width, height):
        target = target.astype(np.double)
        # print("convert:", target, width, height)
        target[:, 0:4:2] /= width
        # print("convert2:", target, width, height)
        target[:, 1:4:2] /= height
        # print("convert3:", target, width, height)
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Polypcoco_anchorfree(Dataset):
    NUM_CLASSES = 1
    CAT_LIST = [-1, 1]

    # CAT_LIST = [-1, 1, 2]
    def __init__(self,
                 base_dir,
                 split='train',
                 bbox_transform=None,
                 target_transform=PolypAnnotationTransform()
                 ):
        super(Polypcoco_anchorfree, self).__init__()
        # print("="*20, "start init", "="*20)
        #print(base_dir)
        anno_path = base_dir + "/large_dataset/polyp_large_train_annots.json"
        self.annos = json.load(open(anno_path, 'r'))
        # #print("anno_files: ", anno_files)
        # ids_file = os.path.join(base_dir, '/large_dataset/{}_ids.pth'.format(split))
        assert len(self.annos) > 0
        self.bbox_transform = bbox_transform
        self.target_transform = target_transform

        # assert cfg.NUM_CLASS == len(self.CAT_LIST), 'Number of class does not match with number of category'

        # minimum mask size
        #self.mask_min_size = 100
        self.img_dir = os.path.join(base_dir, "large_dataset")
        self.split = split
        sys.stdout = open(os.devnull, "w")
        #self.coco_anno_list = COCO(anno_files)
        # print("*"*100)
        # print(annos)
        # cats = self.coco_anno_list.loadCats(self.coco_anno_list.getCatIds())
        # nms = [cat['name'] for cat in cats]
        # print('COCO categories: \n{}\n'.format(' '.join(nms)))
        #print("*"*100)
        sys.stdout = sys.__stdout__
        #self.index_to_coco_anno_index = []
        self.tsf = FSTrans(600, 1000)
        #self.coco_mask = mask
        # if os.path.exists(ids_file):
        #     cache_file = torch.load(ids_file)
        #     self.ids_list = cache_file['id_list']
        #     self.index_to_coco_anno_index = cache_file['index_to_coco_anno_index']
        #     assert self.CAT_LIST == cache_file['cat_list'], "category list differ from what in {} , delete cache and " \
        #                                                     "regenerate it".format(ids_file)
           # for index, anno in enumerate(self.coco_anno_list):
        # for key_index, anno in enumerate(annos):
        #     print(key_index, anno)
#                self.index_to_coco_anno_index.append([index, key_index])

#        self.ids_list, self.index_to_coco_anno_index = self._preprocess_anno_list(ids_file)
        # print("="*20, "end init", "="*20)

    def __getitem__(self, index):
        # print("test get item")
        #print("index: ", index, "self.annos[index]: ", self.annos[index])
        self._img_path = self.annos[index]['filename']
        _img = imageio.imread(os.path.join(self.img_dir, self._img_path))
        gt_bboxs = self.annos[index]['gt_bboxes']
        sample = {'image': _img, 'label': gt_bboxs}
        #augmented_mask = np.array(sample["label"])
        gt_image = sample['image']
        #gt_bboxs = self._mask_to_bbox(augmented_mask)
        height, width, channel = gt_image.shape

        # print("orig gt_bboxs:", gt_bboxs)
        img_info = [width, height]
        # print("before transpose: ", gt_image.shape)
        gt_image = gt_image.transpose(2, 0, 1)
        # print("polyp before transformation: ", gt_image.shape, gt_image)
        # print("after transpose: ", gt_image.shape)
        #print(gt_bboxs)
        #assert gt_bboxs.shape[0] > 0, 'Empty ground truth bounding box on index {}'.format(index)

        boxes = torch.FloatTensor(gt_bboxs)

        boxes = boxes[:, :4]
        # print("before swtich: ", boxes)
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]
        #boxes = boxes.tolist()
        # print("switched: ", boxes)
        # boxes[:, [0, 2]] = boxes[:, [2, 0]]
        # boxes[:, [1, 2]] = boxes[:, [2, 1]]
        # boxes[:, [2, 3]] = boxes[:, [3, 2]]
        # print("polyp before transfomation bbox: ", boxes)
        labels = self.annos[index]['gt_labels']
        #print("image shape: ", gt_image.shape, "box: ", boxes, "label: ", labels)
        if self.split == 'train':

            # print("input:", gt_image.shape, boxes.shape, labels.shape)
            # print("intput box:", boxes)
            img, bbox, label, scale = self.tsf((gt_image, boxes, labels))
            # print("out", img.shape, bbox.shape, label.shape, scale)
            # print("polyp after transformation: ", img.shape, img)
            # print("polyp after transformation bbox: ", bbox.shape, bbox)
            # print(bbox, label)
            # return torch.from_numpy(np.array(gt_image)).float(), gt_targets, img_info
            return img.tolist().copy(), bbox.tolist().copy(), label.copy(), scale
        else:
            img = preprocess(gt_image)
            difficult = np.asarray([0] * len(labels))
            return img, gt_image.shape[1:], boxes, labels, difficult

    # def _mask_to_bbox(self, coco_mask):
    #     class_id = np.unique(coco_mask)
    #     bboxs = []
    #     for i in class_id:
    #         binary_mask = np.zeros(coco_mask.shape[:2], dtype=np.uint8)
    #         if i == 0:
    #             continue
    #         binary_mask[coco_mask == i] = 1
    #         contours = measure.find_contours(binary_mask, 0.5)
    #         for contour in contours:
    #             contour = np.flip(contour, axis=1)
    #             min_x = np.min(contour[:, 0])
    #             min_y = np.min(contour[:, 1])
    #             max_x = np.max(contour[:, 0])
    #             max_y = np.max(contour[:, 1])
    #             area = (max_x - min_x) * (max_y - min_y)
    #             if area < self.mask_min_size:
    #                 continue
    #             bbox = [min_x, min_y, max_x, max_y, i]
    #             bboxs.append(bbox)
    #
    #     return np.array(bboxs, dtype=np.int)

    # def _make_img_gt_point_pair(self, index):
    #     coco_anno_list_index, anno_id_index = self.index_to_coco_anno_index[index]
    #     coco = self.coco_anno_list[coco_anno_list_index]
    #     img_id = self.ids_list[coco_anno_list_index][anno_id_index]
    #     img_metadata = coco.loadImgs(img_id)[0]
    #     path = img_metadata['file_name']
    #     _img = imageio.imread(os.path.join(self.img_dir, path))
    #     # print(os.path.join(self.img_dir, path))
    #     cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
    #     _target = Image.fromarray(self._gen_seg_mask(
    #         cocotarget, img_metadata['height'], img_metadata['width']))
    #
    #     return _img, _target

    # def _preprocess_anno_list(self, ids_file):
    #     print("Preprocessing mask, this will take a while. " +
    #           "But don't worry, it only run once for each split.")
    #     tbar = trange(len(self.index_to_coco_anno_index))
    #     new_index_to_coco_anno_index = []
    #     tbar_counter = 1
    #
    #     ids_list = []
    #     tbar_iter = iter(tbar)
    #     for anno_index, coco_anno in enumerate(self.coco_anno_list):
    #         ids = list(coco_anno.imgs.keys())
    #         new_ids = []
    #         id_counter = 0
    #         for i, id in enumerate(ids):
    #             tbar_iter.__next__()
    #             img_id = id
    #             cocotarget = coco_anno.loadAnns(coco_anno.getAnnIds(imgIds=img_id))
    #             img_metadata = coco_anno.loadImgs(img_id)[0]
    #             mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
    #                                       img_metadata['width'])
    #             # more than mask_min_size pixels
    #             if (mask > 0).sum() > self.mask_min_size:
    #                 new_ids.append(img_id)
    #                 new_index_to_coco_anno_index.append([anno_index, id_counter])
    #                 id_counter += 1
    #
    #             tbar.set_description(
    #                 'Doing: {}/{}, got {} qualified images'.format(tbar_counter, len(self.index_to_coco_anno_index),
    #                                                                len(new_index_to_coco_anno_index)))
    #             tbar_counter += 1
    #         ids_list.append(new_ids)
    #     print('Found number of qualified images: ', len(new_index_to_coco_anno_index))
    #     id_index_to_save = {
    #         'id_list': ids_list,
    #         'index_to_coco_anno_index': new_index_to_coco_anno_index,
    #         'cat_list': self.CAT_LIST
    #     }
    #     torch.save(id_index_to_save, ids_file)
    #     return ids_list, new_index_to_coco_anno_index

    # def _gen_seg_mask(self, target, h, w):
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     coco_mask = self.coco_mask
    #     for instance in target:
    #         rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
    #         m = coco_mask.decode(rle)
    #         cat = instance['category_id']
    #         if cat in self.CAT_LIST:
    #             c = self.CAT_LIST.index(cat)
    #         else:
    #             continue
    #         if len(m.shape) < 3:
    #             mask[:, :] += (mask == 0) * (m * c)
    #         else:
    #             mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
    #     return mask

    def __len__(self):
        return len(self.annos)
