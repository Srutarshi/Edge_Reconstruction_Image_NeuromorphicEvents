import os
import random

def split_data(split_train, split_test, ismain = False):
	root = os.path.join(os.getcwd(), 'data/images' if not ismain else 'images')
	if 'Edge_Reconstruction' not in root:
		root = os.path.join(root[0], 'Edge_Reconstruction/', root[1:])
	files = []
	for r,d,f in os.walk(root):
		for file in f:
			name_abs = os.path.join(r, file)
			name_rel = name_abs.rpartition('/')[2]
			name_part = name_rel.partition('.')
			if name_part[0] != '000000' and name_part[2] == 'jpg':
				files.append(name_abs[len(root)+1:])

	num_train = int(0.7 * len(files))
	num_test = int(0.85 * len(files))
	random.shuffle(files)

	train_files = files[:num_train]
	val_files = files[num_train:num_test]
	test_files = files[num_test:]

	root = os.path.join(os.getcwd(), 'data' if not ismain else '')
	if 'Edge_Reconstruction' not in root:
		root = os.path.join(root[0], 'Edge_Reconstruction/', root[1:])
	with open(os.path.join(root, 'train_files.txt'), 'w+') as f:
		for item in train_files:
			f.write('%s\n' % item)

	with open(os.path.join(root, 'val_files.txt'), 'w+') as f:
		for item in val_files:
			f.write('%s\n' % item)

	with open(os.path.join(root, 'test_files.txt'), 'w+') as f:
		for item in test_files:
			f.write('%s\n' % item)

if __name__ == '__main__':
	split_data(0.7, 0.3, ismain = True)

###