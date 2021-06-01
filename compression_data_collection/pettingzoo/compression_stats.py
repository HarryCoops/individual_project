from PIL import Image
import os
import lz4.frame 
import numpy as np
import csv

def get_compressibility(f):
	im = Image.open(f)
	size = []
	w, h = im._size
	original_size = w * h * 3
	try:
		while True:
			frame = im.seek(im.tell()+1)
			numpy_frame = np.array(im)
			size.append(len(lz4.frame.compress(numpy_frame)))
	except EOFError:
		return original_size, sum(size) / len(size)


if __name__ == "__main__":
	compression_stats =[]
	for f_name in os.listdir("."):
		if f_name.endswith(".gif"):
			compression_stats.append((f_name, ) + get_compressibility(f_name))
	headings = ("game", "original_size", "avg_compressed_size")
	with open("results.csv", "w") as f:
		writer = csv.writer(f)
		writer.writerow(headings)
		writer.writerows(compression_stats)
