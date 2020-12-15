from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import numpy as np
import argparse
from PIL import Image

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--image_dir', type=str, required=True)
    argparser.add_argument('--image_list_file', type=str, required=True)
    argparser.add_argument('--ssim_matrix_file', type=str, required=True)
    argparser.add_argument('--mse_matrix_file', type=str, required=True)
    argparser.add_argument('--image_symlink_dir', type=str, required=True)
    args = argparser.parse_args()

    image_dir = Path(args.image_dir)

    images = []
    for image in image_dir.glob('**/*'):
        if image.is_dir():
            continue
        images.append(image)

    # Save image list so we can identify images later
    images.sort()
    image_list_file = open(args.image_list_file, 'w')
    zfill_len = len(str(len(images)))
    for i in range(len(images)):
        image_list_file.write(f'{i}, {images[i]}\n')
        symlink_dir = Path(args.image_symlink_dir) / Path(str(i).zfill(zfill_len))
        symlink_dir.mkdir(exist_ok = True, parents=True)
        symlink_target = Path('../'*len(symlink_dir.parts)) / images[i]
        (symlink_dir / images[i].name).symlink_to(symlink_target)

    image_list_file.close()

    ssim_matrix_file = open(args.ssim_matrix_file, 'w')
    mse_matrix_file = open(args.mse_matrix_file, 'w')
    for i in range(len(images)):
        i_img = np.array(Image.open(images[i]))
        for j in range(i+1, len(images)):
            j_img = np.array(Image.open(images[j]))
            ssim_matrix_file.write(str(ssim(i_img, j_img, multichannel=True))+',')
            mse_matrix_file.write(str(mean_squared_error(i_img, j_img))+',')
        ssim_matrix_file.write('\n')
        mse_matrix_file.write('\n')
    ssim_matrix_file.close()
    mse_matrix_file.close()

if __name__ == '__main__':
    main()
