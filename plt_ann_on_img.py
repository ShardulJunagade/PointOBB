# plot the annotation file and image path and plot the boxes on img
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

METAINFO = {
	'classes': ('CFCBK', 'FCBK', 'Zigzag'),
	'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
}

add_score = False


def draw_annotations(image, ann_file, img_size, is_ann_normalized, ann_format='dota', is_label_number=False):
	"""Draws rotated bounding boxes from a DOTA or Supervision annotation file using specified colors."""
	img = image.copy()
	if not os.path.exists(ann_file):
		print(f"Annotation file {ann_file} not found.")
		return img
	class_to_color = dict(zip(METAINFO['classes'], METAINFO['palette']))
	with open(ann_file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		values = line.strip().split()
		if len(values) < 10:
			print(f"Invalid annotation line: {line} in {ann_file}")
			continue
		if ann_format == 'dota':
			points = np.array(list(map(float, values[:8]))).reshape((4, 2))
			label = values[8]
			score = values[9]
		elif ann_format == 'supervision':
			label = values[0]
			points = np.array(list(map(float, values[1:9]))).reshape((4, 2))
			score = values[9]
		else:
			print(f"Unsupported annotation format: {ann_format}")
			continue
		if is_label_number:
			label = METAINFO['classes'][int(label)]
		if is_ann_normalized:
			points *= img_size
		points = points.astype(int)
		color = class_to_color.get(label, (255, 255, 255))
		if add_score:
			label = label + ':' +  score
		thickness = 1
		cv2.polylines(img, [points], isClosed=True, color=color, thickness=thickness)
		cv2.putText(img, label, (points[0][0], points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return img


def annotate_directory(image_dir, ann_dir, out_dir, img_size, is_ann_normalized, ann_format='dota', is_label_number=False, save=True, plot=False): 
	"""Draws rotated bounding boxes from DOTA or Supervision annotation files on images in a directory."""
	if not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)
	image_files = sorted(os.listdir(image_dir))
	num_images = len(image_files)
	for i, image_name in enumerate(image_files[:10]):
		print(f'Processing image {i + 1}/{num_images}...')
		image_path = os.path.join(image_dir, image_name)
		if image_name.lower().endswith('.tif') or image_name.lower().endswith('.tiff'):
			image = np.array(Image.open(image_path).convert('RGB'))  # Ensure 3-channel RGB
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
		else:
			image = cv2.imread(image_path)
		if image is None:
			print(f"Image {image_path} not found.")
			continue
		ann_file = os.path.join(ann_dir, f'{os.path.splitext(image_name)[0]}.txt')
		annotated_img = draw_annotations(image, ann_file, img_size, is_ann_normalized, ann_format, is_label_number)
		if save:
			out_path = os.path.join(out_dir, image_name)
			success = cv2.imwrite(out_path, annotated_img)
			if not success:
				print(f"Failed to save annotated image to {out_path}")
		if plot:
			plt.figure(figsize=(10, 10))
			plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
			plt.axis('off')
			plt.show()


image_dir = '/home/patel_zeel/kiln_compass_24/data/bihar/images'
ann_dir = 'results-resnet50/train_combined_test_5states/bihar/annfiles'
out_dir = 'results-resnet50/train_combined_test_5states/bihar/annotated_images'
img_size = 640
is_ann_normalized = True
ann_format = 'supervision'
is_label_number = True

annotate_directory(image_dir, ann_dir, out_dir, img_size, is_ann_normalized, ann_format, is_label_number, save=True, plot=True)
