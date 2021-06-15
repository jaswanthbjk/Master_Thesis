from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os
import cv2
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from pascal_voc_writer import Writer

image_folder_path = '/scratch/jbandl2s/acdc/rgb_anon/'
weathers = ['fog', 'night', 'rain', 'snow']
sources = ['train_ref']

image_file_paths = []

for weather in weathers:
    for source in sources:
        for folder in os.listdir(os.path.join(image_folder_path, weather,
                                              source)):
            image_file_paths += glob.glob(os.path.join(image_folder_path,
                                                       weather, source,
                                                       folder) + '/*.png')

mask_to_class = {9: 'Traffic_light',
                 11: 'Traffic_sign',
                 12: 'Traffic_sign',
                 0: 'Pedestrian',
                 2: 'Car',
                 7: 'Truck',
                 5: 'Bus',
                 6: 'Train',
                 3: 'Motorcycle',
                 1: 'Bicycle'}
object_of_interest = list(mask_to_class.keys())


def perform_detection(image_path, cfg, thresh=0.45):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    im = cv2.imread(image_path)
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    pred_classes = outputs['instances'].pred_classes.cpu().tolist()
    boxes = list(outputs["instances"].pred_boxes)
    return pred_classes, boxes


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")

for path_count in range(len(image_file_paths)):
    path = image_file_paths[path_count]
    pred_classes, boxes = perform_detection(path, cfg)
    writer = Writer(path, 1920, 1080)
    for i, obj in enumerate(pred_classes):
        if obj not in object_of_interest:
            continue
        writer.addObject(mask_to_class[obj], boxes[i][0].item(),
                         boxes[i][1].item(), boxes[i][2].item(),
                         boxes[i][3].item())
    label_path = path.replace('png', 'xml')
    writer.save(path)
