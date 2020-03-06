import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import trange
import os
import glob
from pycocotools.coco import COCO
from pycocotools import mask
from PIL import Image, ImageFile
from skimage import measure
import imageio
import sys
# from torchvision import transforms
# from dataloader.dataloader_utils import decode_segmap
# import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PolypAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        pass

    def __call__(self, target, width, height):
        target = target.astype(np.double)
        #print("convert:", target, width, height)
        target[:, 0:4:2] /= width
        #print("convert2:", target, width, height)
        target[:, 1:4:2] /= height
        #print("convert3:", target, width, height)
        return target  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class Polypcoco_anchorfree(Dataset):
    NUM_CLASSES = 1
    CAT_LIST = [-1, 1]

    # CAT_LIST = [-1, 1, 2]
    def __init__(self,
                 cfg,
                 base_dir,
                 split='train',
                 bbox_transform=None,
                 target_transform=PolypAnnotationTransform()
                 ):
        super(Polypcoco_anchorfree, self).__init__()

        anno_files = glob.glob(os.path.join(base_dir, "annos/{}".format(split), '*.json'))
        ids_file = os.path.join(base_dir, 'annos/{}_ids.pth'.format(split))
        assert len(anno_files) > 0
        self.bbox_transform = bbox_transform
        self.target_transform = target_transform

        #assert cfg.NUM_CLASS == len(self.CAT_LIST), 'Number of class does not match with number of category'

        # minimum mask size
        self.mask_min_size = 100
        self.img_dir = os.path.join(base_dir, "images/")
        self.split = split
        sys.stdout = open(os.devnull, "w")
        self.coco_anno_list = [COCO(anno) for anno in anno_files]
        sys.stdout = sys.__stdout__
        self.index_to_coco_anno_index = []

        self.coco_mask = mask
        if os.path.exists(ids_file):
            cache_file = torch.load(ids_file)
            self.ids_list = cache_file['id_list']
            self.index_to_coco_anno_index = cache_file['index_to_coco_anno_index']
            assert self.CAT_LIST == cache_file['cat_list'], "category list differ from what in {} , delete cache and " \
                                                            "regenerate it".format(ids_file)
        else:
            for index, anno in enumerate(self.coco_anno_list):
                for key_index, _ in enumerate(list(anno.imgs.keys())):
                    self.index_to_coco_anno_index.append([index, key_index])

            self.ids_list, self.index_to_coco_anno_index = self._preprocess_anno_list(ids_file)

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        augmented_mask = np.array(sample["label"])
        gt_image = sample['image']
        gt_bboxs = self._mask_to_bbox(augmented_mask)
        height, width, channel = gt_image.shape
        #print(gt_image.shape)
        #print("orig gt_bboxs:", gt_bboxs)
        img_info = [width, height]
        assert gt_bboxs.shape[0] > 0, 'Empty ground truth bounding box on index {}'.format(index)
        if self.target_transform is not None:
            gt_bboxs = self.target_transform(gt_bboxs, width, height)
        #print("orig2 gt_bboxs:", gt_bboxs)
        if self.bbox_transform is not None:
            #if self.split == 'test':
                #print("gt:", gt_bboxs)
                #print("gt2:",gt_bboxs[:, :4], gt_bboxs[:, 4])
            if self.split == 'train':
                #print("orig3 gt_bboxs:", gt_bboxs)
                #print(gt_bboxs[:, :4], gt_bboxs[:, 4])
                gt_image, boxes, labels = self.bbox_transform(gt_image, gt_bboxs[:, :4], gt_bboxs[:, 4])
                #print("orig4 gt_bboxs:", gt_bboxs)
                #print("before trans:", gt_image.shape)
                gt_image = gt_image.transpose(2 , 0 , 1) # convert 300, 300, 3 to 3, 300, 300
                #print("after trans:", gt_image.shape)
            else:
                gt_image, (boxes, labels) = self.bbox_transform(gt_image, (gt_bboxs[:, :4], gt_bboxs[:, 4]))
                #gt_image = gt_image.permute(2, 0, 1)
            #print("before trans:", gt_bboxs, "\n", channel, height, width,"\n", "after trans:", boxes)
            #gt_image, gt_targets = self.bbox_transform(gt_image, gt_bboxs)
            gt_bboxs = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        gt_targets = torch.FloatTensor(gt_bboxs)
        #print("targettype:", gt_targets.shape)
        #print("final:", gt_targets)
        return torch.from_numpy(np.array(gt_image)).float(), gt_targets, img_info

    def _mask_to_bbox(self, coco_mask):
        class_id = np.unique(coco_mask)
        bboxs = []
        for i in class_id:
            binary_mask = np.zeros(coco_mask.shape[:2], dtype=np.uint8)
            if i == 0:
                continue
            binary_mask[coco_mask == i] = 1
            contours = measure.find_contours(binary_mask, 0.5)
            for contour in contours:
                contour = np.flip(contour, axis=1)
                min_x = np.min(contour[:, 0])
                min_y = np.min(contour[:, 1])
                max_x = np.max(contour[:, 0])
                max_y = np.max(contour[:, 1])
                area = (max_x - min_x) * (max_y - min_y)
                if area < self.mask_min_size:
                    continue
                bbox = [min_x, min_y, max_x, max_y, i]
                bboxs.append(bbox)

        return np.array(bboxs, dtype=np.int)

    def _make_img_gt_point_pair(self, index):
        coco_anno_list_index, anno_id_index = self.index_to_coco_anno_index[index]
        coco = self.coco_anno_list[coco_anno_list_index]
        img_id = self.ids_list[coco_anno_list_index][anno_id_index]
        img_metadata = coco.loadImgs(img_id)[0]
        path = img_metadata['file_name']
        _img = imageio.imread(os.path.join(self.img_dir, path))
        #print(os.path.join(self.img_dir, path))
        cocotarget = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        _target = Image.fromarray(self._gen_seg_mask(
            cocotarget, img_metadata['height'], img_metadata['width']))

        return _img, _target

    def _preprocess_anno_list(self, ids_file):
        print("Preprocessing mask, this will take a while. " +
              "But don't worry, it only run once for each split.")
        tbar = trange(len(self.index_to_coco_anno_index))
        new_index_to_coco_anno_index = []
        tbar_counter = 1

        ids_list = []
        tbar_iter = iter(tbar)
        for anno_index, coco_anno in enumerate(self.coco_anno_list):
            ids = list(coco_anno.imgs.keys())
            new_ids = []
            id_counter = 0
            for i, id in enumerate(ids):
                tbar_iter.__next__()
                img_id = id
                cocotarget = coco_anno.loadAnns(coco_anno.getAnnIds(imgIds=img_id))
                img_metadata = coco_anno.loadImgs(img_id)[0]
                mask = self._gen_seg_mask(cocotarget, img_metadata['height'],
                                          img_metadata['width'])
                # more than mask_min_size pixels
                if (mask > 0).sum() > self.mask_min_size:
                    new_ids.append(img_id)
                    new_index_to_coco_anno_index.append([anno_index, id_counter])
                    id_counter += 1

                tbar.set_description(
                    'Doing: {}/{}, got {} qualified images'.format(tbar_counter, len(self.index_to_coco_anno_index),
                                                                   len(new_index_to_coco_anno_index)))
                tbar_counter += 1
            ids_list.append(new_ids)
        print('Found number of qualified images: ', len(new_index_to_coco_anno_index))
        id_index_to_save = {
            'id_list': ids_list,
            'index_to_coco_anno_index': new_index_to_coco_anno_index,
            'cat_list': self.CAT_LIST
        }
        torch.save(id_index_to_save, ids_file)
        return ids_list, new_index_to_coco_anno_index

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        coco_mask = self.coco_mask
        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        return mask

    def __len__(self):
        return len(self.index_to_coco_anno_index)

if __name__ == "__main__":

    from dataloader.dataloader_utils import decode_segmap
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt
    import argparse
    from config import cfg
    import cv2
    from model.anchor_free import detection_collate_anchorfree
    from model.anchor_free.augmentations import SSDAugmentation
    from utils.timer import Timer

    timer = Timer()
    bbox_transform = SSDAugmentation(size=500, mean=(0, 0, 0))

    # bbox_transform = None
    coco_val = Polypcoco_anchorfree(cfg, "/home/dwang/data0/SynologyDrive/dataset/xiangya",
                                    bbox_transform=bbox_transform, split='train')

    # dataloader = DataLoader(coco_val, batch_size=1, shuffle=True, num_workers=0,collate_fn=detection_collate_anchorfree)
    dataloader = DataLoader(coco_val, batch_size=1, shuffle=True, num_workers=0)
    # rgb
    color = [(255, 0, 0),  # ren,
             (0, 255, 0)  # green,
             ]
    timer.tic()
    for ii, sample in enumerate(dataloader):
        print('data loader time {}'.format(timer.toc()))
        images = sample[0]
        targets = sample[1]
        # augmented_mask = sample[2]
        for i in range(images.shape[0]):
            image = images[i].permute(1, 2, 0).numpy().astype(np.uint8).copy()
            height, width, channel = image.shape
            target = targets[i].numpy()
            target[:, 0:4:2] *= width
            target[:, 1:4:2] *= height
            for t in target:
                pt1 = tuple([int(t[0]), int(t[1])])
                pt2 = tuple([int(t[2]), int(t[3])])
                cls = int(t[4])
                cv2.rectangle(image, pt1, pt2, color[cls - 1], thickness=2)
            # tmp=np.array(augmented_mask[i].numpy()).astype(np.uint8)
            # segmap = decode_segmap(tmp, dataset='coco')
            plt.figure(figsize=(10, 10))
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image)
        # plt.subplot(212)
        # plt.imshow(segmap)

        timer.tic()

        if ii == 20:
            break

    plt.show(block=True)
