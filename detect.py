import argparse
import time
from sys import platform
import pdb
import os
import cv2
from models import *
from utils.datasets import *
from utils.utils import *
from utils import torch_utils


def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False,
        sample_second=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size, sample_second=sample_second)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap, frame) in enumerate(dataloader):
        person_flag = False
        t = time.time()
        if dataloader.mode == "video":
            save_path = str(Path(output) / Path(path).name) + "_out"
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
        else:
            save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred, _ = model(img)
        detections = non_max_suppression(pred, conf_thres, nms_thres)[0]

        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections

            for *xyxy, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    if classes[int(cls)] == "person":
                        person_flag = True
                        if dataloader.mode == "video":
                            # pdb.set_trace()
                            with open(os.path.join(save_path, str(frame)) + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                        else:
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))
                        # Add bbox to the image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=(0, 0, 255))

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save generated image with detections
            if dataloader.mode == 'video':
                if person_flag:
                    cv2.imwrite(os.path.join(save_path, str(frame)) + '.png', im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
                    vid_writer.write(im0)

            else:
                if person_flag:
                    cv2.imwrite(save_path, im0)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.pt', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--save-images', action='store_true', help='Save image result')
    parser.add_argument('--sample-second', action='store_true', help='Take one frame per second for video')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres,
            save_images=opt.save_images,
            save_txt=True,
            sample_second=opt.sample_second
        )

# python3 detect.py --weights weights/best2.pt --img-size 416 \
#                   --images /home/chrizandr/sports/detection_exp/annotated --data-cfg data/soccer.data \
#                   --cfg cfg/soccer2.cfg --save-images
# python3 detect.py --weights weights/best.pt --images ~/sports/detection_exp/images/ --save-images
# python3 detect.py --weights weights/yolov3-spp.pt --images ~/sports/
