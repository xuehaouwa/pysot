from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import math
from gv_tools.util.visual import draw_regions
import argparse
from gv_tools.tracking.tracking_region import TrackingRegion
from gv_tools.util.text import write_into_region
import cv2
import torch
import numpy as np
from detection.body_detector import BodyDetector

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo')
parser.add_argument('--config', type=str, help='config file')
parser.add_argument('--snapshot', type=str, help='model name')
parser.add_argument('-m', '--model_path', type=str, default="/media/haoxue/WD/ssd_model_zoo/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb", help="trained model path from model zoo")
parser.add_argument('--video_name', default='', type=str, help='videos or image files')
args = parser.parse_args()


class MSTracker:
    def __init__(self, model, tracker_id):
        self.tracker = build_tracker(model=model)
        self.id = tracker_id

    def update(self, frame):
        outputs = self.tracker.track(frame)
        bbox = list(map(int, outputs['bbox']))
        score = outputs["best_score"]
        track_region = TrackingRegion(left=bbox[0], right=bbox[0] + bbox[2], top=bbox[1], bottom=bbox[1] + bbox[3])
        track_region.data["sot_score"] = score
        track_region.data["sot_id"] = self.id

        return track_region, score

    def tracker_init(self, frame, init_region: TrackingRegion):
        init_rect = (init_region.left, init_region.top, init_region.width, init_region.height)
        self.tracker.init(frame, init_rect)


def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(args.snapshot,
                                     map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    cap = cv2.VideoCapture(args.video_name)
    body_detector = BodyDetector()
    body_detector.load_model(path_to_model=args.model_path)
    first_frame_with_detection = False
    updating_frame = False
    sot_trackers = {}

    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        img_height, img_width, _ = np.shape(frame)
        regions, _, _ = body_detector.process(frame)

        if len(regions) > 0 and not first_frame_with_detection:
            first_frame_with_detection = True

        if first_frame_with_detection and not updating_frame:
            for r in regions:
                counter += 1
                sot_trackers[counter] = MSTracker(model=model, tracker_id=counter)
                sot_trackers[counter].tracker_init(frame, r)

            updating_frame = True
            print(f"Init number of MSTracker: {counter}")
            continue

        if updating_frame:
            current_frame_sot_regions = []
            for tracker_id in sot_trackers.keys():
                sot_region, sot_score = sot_trackers[tracker_id].update(frame)
                if sot_score > 0.5:
                    current_frame_sot_regions.append(sot_region)

            current_detected_regions = regions

            # compare SOT region and detected region to decide whether fire up a new MSTracker
            for d_region in current_detected_regions:
                new_tracker = True
                for sot_region in current_frame_sot_regions:
                    distance = math.sqrt((d_region.x - sot_region.x) ** 2 + (d_region.y - sot_region.y) ** 2)
                    if distance < 200:
                        new_tracker = False
                        break
                if new_tracker:
                    counter += 1
                    sot_trackers[counter] = MSTracker(model=model, tracker_id=counter)
                    sot_trackers[counter].tracker_init(frame, d_region)
                    print(f"New Tracker: {counter}")

            # display
            # displayed = draw_regions(frame, current_frame_sot_regions, color=(0, 255, 0))
            # displayed = draw_regions(displayed, current_detected_regions)
            for r in current_frame_sot_regions:
                t_id = r.data["sot_id"]
                frame = write_into_region(frame, str(t_id), r, show_region_outline=True)
            cv2.putText(frame, f"MSTracker: {len(sot_trackers.keys())}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
            cv2.imshow("body", frame)
            cv2.waitKey(1)


if __name__ == '__main__':
    main()
