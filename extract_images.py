from argparse import ArgumentParser
import pickle
from os.path import join
import numpy as np
import struct
from skimage.io import imsave
from skimage.util import img_as_uint, img_as_ubyte

# constants
PIXEL_TYPE = 'h' # each pixel is 'short'
PIXEL_TYPE_NP = np.int16

FRAME_SHAPE = (64, 96)
FRAME_LEN = np.prod(FRAME_SHAPE)
FRAME_LEN_BYTES = FRAME_LEN * struct.calcsize(PIXEL_TYPE)

IMAGE_SHAPE = (60, 89)

def read_image(frame_index, raw_data):
    frame_byte_offset = frame_index * FRAME_LEN_BYTES
    frame_data = raw_data[frame_byte_offset:frame_byte_offset + FRAME_LEN_BYTES]
    frame = np.array(struct.unpack("<" + PIXEL_TYPE*FRAME_LEN, frame_data)).reshape(FRAME_SHAPE).astype(PIXEL_TYPE_NP)
    return frame[2:62, 5:94] # image is 60x89 frame

def create_image_path(out_dir, frame_num, digits=5, ext=".png"):
    image_filename = str(frame_num).zfill(digits) + ext
    return join(out_dir, image_filename)


if __name__ == "__main__":
    args = ArgumentParser(description="Extract images from .dml file and save them to `out_path` folder.\n\t If `save_all_imgs` is 1 save all extracted images as .png files, otherwise save reference image and pickled data.,\n\t Data is saved as python dictionary with keys 'reference_image' and 'images'.")
    args.add_argument("dml_path", help="Path to .dml file.")
    args.add_argument("out_path", help="Path to output folder.")
    args.add_argument("--save_all_imgs", default=0, type=int, help="1 - save all extracted images as .pngs, 0- save only reference image and pickled data")
    args = args.parse_args()

    raw_data = open(args.dml_path, "rb").read()
    assert len(raw_data) % FRAME_LEN == 0
    frame_num = len(raw_data) // FRAME_LEN_BYTES
    print(f"File {args.dml_path} contains {frame_num} frames.")

    reference_image = read_image(0, raw_data)
    imsave(create_image_path(args.out_path, "reference_image"), reference_image)
    original_data = {"reference_image": reference_image, "images": []}

    for i in range(1, frame_num): # 0 - ref. img
        image = read_image(i, raw_data) + reference_image
        original_data["images"].append(image)
        if args.save_all_imgs:
            j = i - 1
            print(f"Saving image {j}.")
            imsave(create_image_path(args.out_path, j), img_as_ubyte(image / np.abs(image).max()))

    with open(create_image_path(args.out_path, "extracted_data", ext=".pck"), "wb") as out_file:
        pickle.dump(original_data, out_file, protocol=4)

