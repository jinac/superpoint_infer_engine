"""
Script to run OpenVINO model
"""
import argparse

import cv2
import os
import numpy as np

import network
from openvino.inference_engine import IECore, IENetwork, IEPlugin
from superpoint_frontend import SuperPointFrontend

class NetWrapper(object):
    def __init__(self, bin_filename, xml_filename,
                 h, w, nms_dist, conf_thresh, nn_thresh):
        self.ie = IECore()
        self.plugin = IEPlugin(device="CPU")

        self.bin_filename = bin_filename
        self.xml_filename = xml_filename
        self.net = self.load_net()
        self.fe = SuperPointFrontend(h=h, w=w,
                                     nms_dist=nms_dist,
                                     conf_thresh=conf_thresh,
                                     nn_thresh=nn_thresh)


    def load_net(self):
        # net = cv2.dnn.readNet(self.bin_filename, self.xml_filename)
        net = IENetwork(model=self.xml_filename,
                        weights=self.bin_filename)
        exec_net = self.ie.load_network(network=net, device_name="CPU")
        return exec_net

    def infer(self, input):
        res = self.net.infer(inputs={'input': input})
        k = res.keys()
        # print(k)
        out = [res[l] for l in k]
        out = self.fe.process_pts(*out)
        return out


def draw_circles(img, pts, color):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for pt in pts:
        cv2.circle(img, (int(pt[0]), int(pt[1])), 1, color, -1)
    return img


def main():
    parser = argparse.ArgumentParser(
        description='Script to convert superpoint model from pytorch to onnx')
    parser.add_argument('bin_filename', help="model bin filepath")
    parser.add_argument('xml_filename', help="model xml filepath")
    parser.add_argument('--nfeatures', default=2000, type=int,
        help="number of top features to sample")
    parser.add_argument('--nms_dist', default=4, type=int,
        help="nms distance")
    parser.add_argument('--confidence', default=0.015, type=float,
        help="confidence threshold for key pt")

    args = parser.parse_args()

    video_filename = '/mnt/f/slam/bedroom.mp4'
    test_img_filename = 'test_img.jpg'
    onnx_filename = 'superpoint_100x100.onnx'
    bin_filename = 'output/superpoint_100x100.bin'
    xml_filename = 'output/superpoint_100x100.xml'

    nfeatures = args.nfeatures
    inlierThreshold = 0.001
    nms_dist = args.nms_dist
    conf_thresh = args.confidence
    nn_thresh = 0.7
    h = 100
    w = 100

    # Setup
    net = NetWrapper(bin_filename, xml_filename,
                     h=h, w=w, nms_dist=nms_dist,
                     conf_thresh=conf_thresh,
                     nn_thresh=nn_thresh)

    cap = cv2.VideoCapture(video_filename)
    # img = cv2.imread(test_img_filename, 0)
    i = 0
    out_vid = cv2.VideoWriter('/mnt/f/slam/output_bedroom.mp4', cv2.VideoWriter_fourcc('M','P','4','V'), 30, (1024, 1024))
    while(cap.isOpened()):
        ret, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        rows, cols = img.shape
        rows = min(rows, cols)
        cols = min(rows, cols)

        img = img[:rows, :cols]
        img = cv2.resize(img, (100, 100))
        img_input = (img.astype('float32') / 255.)
        # cv2.imwrite('test_output.jpg', img)

        pts, desc, _ = net.infer(img_input)
        try:
            o = img.copy()
            o = draw_circles(o, pts, (255, 0, 0))
            o = cv2.resize(o, (1024, 1024))
            # cv2.imwrite('test/{}.jpg'.format(i), o)
            out_vid.write(o)
            i += 1
            # cv2.imshow('frame', o)
        except:
            continue
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out_vid.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()