"""
Script to run superpoint on MegaAI/OAK-1 module
"""
import cv2
import os
import numpy as np

import consts.resource_paths
import depthai
from superpoint_frontend import SuperPointFrontend

class SuperPointWrapper(object):
    def __init__(self, h, w, nms_dist, conf_thresh, nn_thresh, cell=8):
        self.h = h
        self.w = w
        self.cell = cell
        self.hc = int(self.h / self.cell)
        self.wc = int(self.w / self.cell)
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh
        self.fe = SuperPointFrontend(h=h, w=w,
                                     nms_dist=nms_dist,
                                     conf_thresh=conf_thresh,
                                     nn_thresh=nn_thresh)

    def decode_packet(self, nnet_packet):
        raw_data = nnet_packet.get_tensor('semi')
        raw_data.dtype = np.float16
        nn_outputs = nnet_packet.entries()[0]

        output_shapes = [(1,65,self.hc,self.wc), (1,256,self.hc,self.wc)]
        nn_output_list = []
        prev_offset = 0
        for shape, entry in zip(output_shapes, nnet_packet.entries()[0]):
            n_size = len(entry)
            output = raw_data[prev_offset:prev_offset+n_size]
            output = np.reshape(output, shape)

            nn_output_list.append(output)
            prev_offset += n_size

        # print('len output list: ', len(nn_output_list))
        pts, desc = nn_output_list
        pts, desc, heatmap = self.fe.process_pts(pts, desc)
        return pts, desc, heatmap

    def decode_packets(self, nnet_packets):
        kp_desc_list = []
        for nnet_packet in nnet_packets:
            try:
                pts, desc, _ = self.decode_packet(nnet_packet)
                # print(pts.shape)
                kp_desc_list.append((pts, desc))
            except:
                continue

        return kp_desc_list


def main():
    nfeatures = 2000 # Number of keypoints to sample
    inlierThreshold = 0.001
    nms_dist = 4
    conf_thresh = 0.005
    nn_thresh = 0.7
    h = 100
    w = 100
    res = '{}x{}'.format(h, w)

    if not depthai.init_device(consts.resource_paths.device_cmd_fpath):
        raise RuntimeError("Error initializing device. Try to reset it.")

    local_dir = os.getcwd()
    p = depthai.create_pipeline(config={
        "streams": ["previewout", "metaout"],
        "ai": {
            "blob_file": os.path.join(local_dir, "output", "superpoint_{}.blob".format(res)),
            "blob_file_config": os.path.join(local_dir, "output","superpoint_{}.json".format(res)),
            "calc_dist_to_bb": False,
        },
        "app": { "sync_video_meta_stream": True},
        })

    if p is None:
        raise RuntimeError("Error initializing pipelne")

    sp = SuperPointWrapper(h=h, w=w,
                           nms_dist=nms_dist,
                           conf_thresh=conf_thresh,
                           nn_thresh=nn_thresh)
    kp_desc_list = []
    while True:
        nnet_packets, data_packets = p.get_available_nnet_and_data_packets()
        print('nnet_packets: ', len(nnet_packets))
        packets_len = len(nnet_packets) + len(data_packets)
        print('packets_len: ', packets_len)

        kp_desc_list.extend(sp.decode_packets(nnet_packets))

        for nnet_packet, packet in zip(nnet_packets, data_packets):
            if packet.stream_name == 'previewout':
                data = packet.getData()
                data0 = data[0, :, :]
                frame = cv2.cvtColor(data0, cv2.COLOR_GRAY2BGR)

                if len(kp_desc_list) > 0:
                    kps, _ = kp_desc_list.pop(0)
                    for k in kps:
                        print(k[:2])
                        cv2.circle(frame, (int(k[0]), int(k[1])), 3, (0, 0, 255), -1)

                frame = cv2.resize(frame, (240, 240))

                cv2.imshow('previewout', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    del p
    depthai.deinit_device()

if __name__ == '__main__':
    main()