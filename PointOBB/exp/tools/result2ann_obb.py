from pycocotools.coco import COCO
import json
import argparse
from mmdet.core.bbox import bbox_overlaps
import torch


def check(coco, res):
    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]

                assert ori_ann['id'] == ann['ann_id']
                for key in ['bbox', 'segmentation', 'area', ]:
                    assert key in ori_ann, ori_ann
                    assert key in ann, ann
                    assert ori_ann[key] == ann[key], f"{key}\n\t{ori_ann}\n\t{ann}"
    # print("Check function completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ori_ann", help='Path to the original annotation file, e.g., data/coco/resize/annotations/instances_val2017_100x167.json')
    parser.add_argument("--det_file", help='Path to the detection results file, e.g., exp/latest_result.json')
    parser.add_argument("--save_ann", help='Path to save the transformed annotation file, e.g., exp/rr_latest_result.json')
    args = parser.parse_args()

    print(f"Loading original annotations from: {args.ori_ann}")
    coco = COCO(args.ori_ann)
    print(f"Loading detection results from: {args.det_file}")
    res = coco.loadRes(args.det_file)

    iou_sum = 0
    num_sum = 0

    for im_id in coco.imgToAnns:
        if im_id in res.imgToAnns:
            print(f"Processing image ID: {im_id}")
            anns = res.imgToAnns[im_id]
            for ann in anns:
                ori_ann = coco.loadAnns(ann['ann_id'])[0]
                # print(f"Original annotation: {ori_ann}")
                # print(f"Result annotation: {ann}")

                assert ori_ann['id'] == ann['ann_id'], f"{ori_ann} vs {ann}"

                for key in ['image_id', 'category_id', 'iscrowd']:
                    assert ori_ann[key] == ann[key], key

                for key in ['bbox', 'segmentation', 'area', ]:
                    if key == 'bbox':
                        ba = torch.tensor(ori_ann[key]).unsqueeze(0)
                        # print(f"Original bbox tensor: {ba}")

                    ori_ann[key] = ann[key]
                ori_ann['ann_weight'] = ann['score']
                # print(f"Updated annotation: {ori_ann}")

    print("Running check function...")
    check(coco, res)
    print(f"Saving transformed annotations to: {args.save_ann}")
    json.dump(coco.dataset, open(args.save_ann, 'w'))
    print("Annotations saved successfully.")
    print("Done!")
