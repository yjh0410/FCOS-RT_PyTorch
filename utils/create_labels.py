import numpy as np
import torch


def gt_creator(img_size, num_classes, strides, scale_range, targets):
    batch_size = len(targets)
    w = h = img_size
    gt_tensor = []
    
    # empty gt tensor
    for s in strides:
        gt_tensor.append(np.zeros([batch_size, h//s, w//s, num_classes + 4 + 1]))

    # generate gt datas  
    for bi in range(batch_size):
        target = targets[bi]
        bboxes = target['boxes'].tolist()
        labels = target['labels'].tolist()
        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(cls_id)

            # compute the center, width and height
            xc = (x2 + x1) / 2 * w
            yc = (y2 + y1) / 2 * h
            bw = (x2 - x1) * w
            bh = (y2 - y1) * h

            if bw < 1. or bh < 1.:
                # print('A dirty data !!!')
                continue    

            for si, s in enumerate(strides):
                hs, ws = h // s, w // s
                x1_s, x2_s = x1 * ws, x2 * ws
                y1_s, y2_s = y1 * hs, y2 * hs
                xc_s = xc / s
                yc_s = yc / s
                sr = scale_range[si]

                gridx = int(xc_s)
                gridy = int(yc_s)

                # By default, we only consider the 3x3 neighborhood of the center point
                for i in range(gridx - 1, gridx + 2):
                    for j in range(gridy - 1, gridy + 2):
                        if (j >= 0 and j < gt_tensor[si].shape[1]) and (i >= 0 and i < gt_tensor[si].shape[2]):
                            t = j - y1_s
                            b = y2_s - j
                            l = i - x1_s
                            r = x2_s - i
                            if min(t, b, l, r) > 0:
                                if max(t, b, l, r) >= (sr[0]/s) and max(t, b, l, r) < (sr[1]/s):
                                    gt_tensor[si][bi, j, i, cls_id] = 1.0
                                    gt_tensor[si][bi, j, i, num_classes:num_classes + 4] = np.array([x1, y1, x2, y2])
                                    gt_tensor[si][bi, j, i, num_classes + 4] = np.sqrt(min(l, r) / max(l, r) * \
                                                                                        min(t, b) / max(t, b))
        
    gt_tensor = [gt.reshape(batch_size, -1, num_classes + 4 + 1) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, axis=1)

    return torch.from_numpy(gt_tensor).float()


if __name__ == "__main__":
    pass