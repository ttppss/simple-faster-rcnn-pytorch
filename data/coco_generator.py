def coco_generator(image_path):
    images = []
    for im_path in image_path:
        im = imageio.imread(im_path)
        # image_nbr = re.findall(r"[0-9]+", im_path)
        im_path = str(im_path)
        image_nbr = im_path[(im_path.rfind('/') + 1): im_path.rfind('.')]
        last_slash_pos = im_path.rfind('/')
        without_last_slash = im_path[:last_slash_pos]
        second_last_slash_pos = without_last_slash.rfind('/')
        without_second_last_slash = im_path[:second_last_slash_pos]
        third_last_slash_pos = without_second_last_slash.rfind('/')
        file_name = im_path[(third_last_slash_pos + 1):]
        image_info = {
            "coco_url": "",
            "date_captured": "",
            "flickr_url": "",
            "license": 0,
            "id": image_nbr,
            "file_name": file_name,
            "height": im.shape[0],
            "width": im.shape[1]
        }
        images.append(image_info)

        ground_truth_binary_mask = np.array(im)
        ground_truth_binary_mask_1 = ground_truth_binary_mask.copy()

        # replace 255 with 1 in the data
        ground_truth_binary_mask_1[ground_truth_binary_mask_1 > 1] = 1
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask_1)
        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        # print(image_nbr, ground_truth_binary_mask_1.shape)
        contours = measure.find_contours(ground_truth_binary_mask_1, 0.5)
        cont = []
        for contour in contours:
            contour = np.flip(contour, axis=1)
            cont.append(contour)

        # get the largest x and y coordinate, and then create bbox.
        x_list = []
        y_list = []
        for i in range(len(cont[0])):
            x_list.append(cont[0][i][0])
            y_list.append(cont[0][i][1])
        x_min = min(x_list)
        x_max = max(x_list)
        y_min = min(y_list)
        y_max = max(y_list)
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        # put everything to a single json file.
        # parent_dir = os.path.dirname(path)
        file_name_with_extention = os.path.basename(im_path)
        image_nbr_for_anno = file_name_with_extention.split('.')[0]
        anno = {
            "category_id": 1,
            "id": 1,
            "image_id": image_nbr_for_anno,
            "iscrowd": 0,
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "bbox": bbox,
        }
        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            anno["segmentation"].append(segmentation)
            annotation.append(anno)
        coco = {"categories": [
            {
                "id": 1,
                "name": "polyp",
                "supercategory": ""
            },
            {
                "id": 2,
                "name": "instrument",
                "supercategory": ""
            }
        ],
            "images": images,
            "annotations": annotation}
    return coco
    print(json.dumps(coco, indent=4))


