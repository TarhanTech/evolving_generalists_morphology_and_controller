import cv2
import os
import argparse
def images_to_video(image_folder, output_video, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Usage
parser = argparse.ArgumentParser(description="Create picture motion videos with a folder of images.")
parser.add_argument("--images_path", type=str, help="Path to the folder where the images are stored")
args = parser.parse_args()

images_to_video(args.images_path, "evolution_video.mp4", fps=15) 