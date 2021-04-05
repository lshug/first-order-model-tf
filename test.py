import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import tensorflow as tf
import numpy as np
from animate import animate
import argparse


parser = argparse.ArgumentParser(description="Test inference using a saved_model")
parser.add_argument("--model", action="store", type=str, default="vox", nargs=1, dest="model", help="model name")
parser.add_argument("--source_image", action="store", dest="source_image", type=str, nargs=1, default="example/source.png", help="source image path")
parser.add_argument("--driving_video", action="store", dest="driving_video", type=str, nargs=1, default="example/driving.mp4", help="driving video path")
parser.add_argument("--output", action="store", type=str, default="example/output", nargs=1, dest="output", help="model name")
parser.add_argument("--relative", dest="relative", action="store_true")
parser.add_argument("--adapt", dest="adapt_movement_scale", action="store_true")
parser.add_argument("--frames", dest="frames", type=int, default=-1, help="number of frames to process")

parser = parser.parse_args()


source_image = imageio.imread(parser.source_image)
source_image = source_image[..., :3]
reader = imageio.get_reader(parser.driving_video)
fps = reader.get_meta_data()["fps"]
reader.close()
driving_video = imageio.mimread(parser.driving_video, memtest=False)
source_image = resize(source_image, (256, 256))[..., :3][None].astype("float32")
source = source_image.astype(np.float32)
if parser.frames != -1:
    driving_video = driving_video[0 : parser.frames]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video][0 : len(driving_video) if len(tf.config.list_physical_devices("GPU")) > 0 else 200]
frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]

kp_detector_loader = tf.saved_model.load("saved_models/" + parser.model + "/kp_detector")
kp_detector_base = kp_detector_loader.signatures["serving_default"]


def kp_detector(l):
    kp_detector_out = kp_detector_base(img=l)
    return kp_detector_out["output_0"], kp_detector_out["output_1"]


generator_loader = tf.saved_model.load("saved_models/" + parser.model + "/generator")
generator_base = generator_loader.signatures["serving_default"]
generator = lambda l: generator_base(source_image=l[0], kp_driving=l[1], kp_driving_jacobian=l[2], kp_source=l[3], kp_source_jacobian=l[4])["output_0"]

process_kp_driving_loader = tf.saved_model.load("saved_models/" + parser.model + "/process_kp_driving")
process_kp_driving_base = process_kp_driving_loader.signatures["serving_default"]


def process_kp_driving(l, m, n, o, p, q, r, s, t):
    process_kp_driving_out = process_kp_driving_base(
        kp_driving=l,
        kp_driving_jacobian=m,
        kp_driving_initial=n,
        kp_driving_initial_jacobian=o,
        kp_source=p,
        kp_source_jacobian=q,
        use_relative_movement=tf.convert_to_tensor(r),
        use_relative_jacobian=tf.convert_to_tensor(s),
        adapt_movement_scale=tf.convert_to_tensor(t),
    )
    return process_kp_driving_out["output_0"], process_kp_driving_out["output_1"]


predictions = animate(source_image, frames, generator, kp_detector, process_kp_driving, 4, parser.relative, parser.adapt_movement_scale)
imageio.mimsave(parser.output + ".saved_model.mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)
