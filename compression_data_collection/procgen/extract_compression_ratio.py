import os
import cv2
import csv
import lz4.frame

# based on https://stackoverflow.com/questions/42163058/how-to-turn-a-video-into-numpy-array
def extract_compression_data(folder):
    sizes = []
    original_size = None
    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            cap = cv2.VideoCapture(folder + "/" + filename)
            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frameCount == 0:
                continue
            fc = 0
            ret = True
            while fc < frameCount and ret:
                ret, im = cap.read()
                if ret:
                    if original_size is None:
                        w, h, d = im.shape
                        original_size = w*h*d
                    sizes.append(len(lz4.frame.compress(im)))
            cap.release()
    return original_size, sum(sizes) / len(sizes)


if __name__ == "__main__":
    compression_data = []
    envs = os.listdir("./videos")
    for env_name in envs:
        if os.path.isdir(f"./videos/{env_name}"):
            compression_data.append(
                (env_name,) + extract_compression_data("videos/" + env_name)
            )
    headings = ("game", "original_size", "avg_compressed_size")
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(headings)
        writer.writerows(compression_data)
    