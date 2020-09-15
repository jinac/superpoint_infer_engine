import numpy as np

# https://github.com/pytorch/pytorch/blob/f064c5aa33483061a48994608d890b968ae53fb5/aten/src/THNN/generic/SpatialGridSamplerBilinear.c#L62
def grid_sample(coarse_desc, samp_pts, mode='bilinear'):
    out = np.zeros((coarse_desc.shape[1], samp_pts.shape[2]))
    n, c, h, w = coarse_desc.shape
    samp_pts = samp_pts.reshape((samp_pts.shape[2:]))

    for i, (ix, iy) in enumerate(samp_pts):
        # Normalize samp_pt x,y from [-1, 1] to [0, h-1], [0, w-1]
        # See https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cpu/GridSamplerKernel.cpp#L283
        ix = ((ix + 1.0) / 2) * w - 0.5
        iy = ((iy + 1.0) / 2) * h - 0.5
        # ix = ((ix + 1.0) / 2) * (w-1)
        # iy = ((iy + 1.0) / 2) * (h-1)

        if mode == 'bilinear':
            # Get neighboring pixels and weights
            x0 = np.floor(ix).astype(int)
            x1 = x0 + 1
            y0 = np.floor(iy).astype(int)
            y1 = y0 + 1

            wa = (x1-ix) * (y1-iy)
            wb = (x1-ix) * (iy-y0)
            wc = (ix-x0) * (y1-iy)
            wd = (ix-x0) * (iy-y0)

            val = wa * coarse_desc[0, :, y0, x0]
            val += wb * coarse_desc[0, :, y1, x0]
            val += wc * coarse_desc[0, :, y0, x1]
            val += wd * coarse_desc[0, :, y1, x1]

        if mode == 'nearest':
            # nearest
            ix_nw = int(np.around(ix))
            iy_nw = int(np.around(iy))

            if ix_nw < w and iy_nw < h:
                val = coarse_desc[0, :, iy_nw, ix_nw]
            else:
                val = np.zeros(c)

        out[:, i] = val

    return out


def reduce_l2(desc):
    dn = np.linalg.norm(desc, ord=2, axis=1) # Compute the norm.
    desc = desc / np.expand_dims(dn, 1) # Divide by norm to normalize.
    return desc


class SuperPointFrontend(object):
    def __init__(self, h, w, nms_dist, conf_thresh, nn_thresh):
        self.h = h
        self.w = w

        self.name = 'SuperPoint'
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh # L2 descriptor distance for good match.
        self.cell = 8 # Size of each output cell. Keep this fixed.
        self.border_remove = 4 # Remove points this close to the border.

        self.Hc = int(self.h / self.cell)
        self.Wc = int(self.w / self.cell)

    def nms_fast(self, in_corners, H, W, dist_thresh):

        grid = np.zeros((H, W)).astype(int) # Track NMS data.
        inds = np.zeros((H, W)).astype(int) # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2,:])
        corners = in_corners[:,inds1]
        rcorners = corners[:2,:].round().astype(int) # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1,i], rcorners[0,i]] = 1
            inds[rcorners[1,i], rcorners[0,i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
          # Account for top and left padding.
            pt = (rc[0]+pad, rc[1]+pad)
            if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
                grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid==-1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds


    def process_pts(self, semi, coarse_desc, sampled=2000):
        coarse_desc = reduce_l2(coarse_desc)

        # --- Process points.
        semi = semi[0, :, :, :]
        dense = np.exp(semi) # Softmax.
        dense = dense / (np.sum(dense, axis=0)+.00001) # Shonwd sum to 1.
        nodust = dense[:-1, :, :] # remove last value of 65 b/c it's a throwaway
        # Reshape to get fnwl resolution heatmap.

        nodust = np.transpose(nodust, [1, 2, 0])
        heatmap = np.reshape(nodust, [self.Hc, self.Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [self.Hc*self.cell, self.Wc*self.cell])
        # prob_map = heatmap/np.sum(np.sum(heatmap))

        xs, ys = np.where(heatmap >= self.conf_thresh) # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None

        pts = np.zeros((3, len(xs))) # Popnwate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, self.h, self.w, dist_thresh=self.nms_dist) # Apply NMS.
        inds = np.argsort(pts[2,:])
        pts = pts[:,inds[::-1]] # Sort by confidence.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (self.h-bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (self.w-bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # pts = pts[:,0:sampled] #we take 2000 keypoints with highest probability from heatmap for our benchmark

        # --- Process descriptor
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
          # Interpolate into descriptor map using 2D point locations.
            samp_pts = pts[:2, :].copy()
            samp_pts[0, :] = (samp_pts[0, :] / (float(self.w)/2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(self.h)/2.)) - 1.
            samp_pts = samp_pts.transpose()
            samp_pts = samp_pts.astype('float32')
            samp_pts = np.reshape(samp_pts, (1, 1, -1, 2))

            desc = grid_sample(coarse_desc, samp_pts)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        return pts.transpose(), desc, heatmap

