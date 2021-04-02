import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import tensorflow as tf
import numpy as np
from animate import animate
import argparse

parser = argparse.ArgumentParser(description="Test inference using tf lite models")
parser.add_argument("--model", action="store", type=str, default="vox", nargs=1, dest="model", help="model name")
parser.add_argument("--source_image", action="store", dest="source_image", type=str, nargs=1, default="example/source.png", help="source image path")
parser.add_argument("--driving_video", action="store", dest="driving_video", type=str, nargs=1, default="example/driving.mp4", help="driving video path")
parser.add_argument("--output", action="store", type=str, default="example/output", nargs=1, dest="output", help="model name")
parser.add_argument("--relative", dest="relative", action="store_true")
parser.add_argument("--adapt", dest="adapt_movement_scale", action="store_true")
parser = parser.parse_args()


source_image = imageio.imread(parser.source_image)
source_image = source_image[..., :3]
reader = imageio.get_reader(parser.driving_video)
fps = reader.get_meta_data()["fps"]
reader.close()
driving_video = imageio.mimread(parser.driving_video, memtest=False)
source_image = resize(source_image, (256, 256))[..., :3][None].astype("float32")
source = source_image.astype(np.float32)
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video][0 : len(driving_video) if len(tf.config.list_physical_devices("GPU")) > 0 else 200]
frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]


kp_detector_interpreter = tf.lite.Interpreter(model_path="tflite/" + parser.model + "/kp_detector.tflite")
kp_detector_input_index = kp_detector_interpreter.get_input_details()[0]["index"]
kp_detector_output_index = kp_detector_interpreter.get_output_details()[0]["index"]


def kp_detector(img):
    kp_detector_interpreter.resize_tensor_input(kp_detector_input_index, img.shape)
    kp_detector_interpreter.allocate_tensors()
    kp_detector_interpreter.set_tensor(kp_detector_input_index, img)
    kp_detector_interpreter.invoke()
    return kp_detector_interpreter.get_tensor(kp_detector_output_index)


generator_interpreter = tf.lite.Interpreter(model_path="tflite/" + parser.model + "/generator.tflite")
source_image_index = [x for x in generator_interpreter.get_input_details() if "source_image" in x["name"]][0]["index"]
generator_kp_driving_index = [x for x in generator_interpreter.get_input_details() if "kp_driving" in x["name"]][0]["index"]
generator_kp_source_index = [x for x in generator_interpreter.get_input_details() if "kp_source" in x["name"]][0]["index"]
generator_output_index = generator_interpreter.get_output_details()[0]["index"]


def generator(inputs):
    generator_interpreter.resize_tensor_input(generator_kp_driving_index, inputs[1].shape)
    generator_interpreter.allocate_tensors()
    generator_interpreter.set_tensor(source_image_index, inputs[0])
    generator_interpreter.set_tensor(generator_kp_driving_index, inputs[1])
    generator_interpreter.set_tensor(generator_kp_source_index, inputs[2])
    generator_interpreter.invoke()
    return generator_interpreter.get_tensor(generator_output_index)


process_kp_driving_interpreter = tf.lite.Interpreter(model_path="tflite/" + parser.model + "/process_kp_driving.tflite")
process_kp_driving_kp_driving_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_driving" in x["name"]][0]["index"]
process_kp_driving_kp_source_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_source" in x["name"]][0]["index"]
process_kp_driving_relative_index = [x for x in process_kp_driving_interpreter.get_input_details() if "relative" in x["name"]][0]["index"]
process_kp_driving_adapt_movement_scale_index = [x for x in process_kp_driving_interpreter.get_input_details() if "adapt" in x["name"]][0]["index"]
process_kp_driving_output_index = process_kp_driving_interpreter.get_output_details()[0]["index"]


def process_kp_driving(kp_driving, kp_source, relative, adapt_movement_scale):
    process_kp_driving_interpreter.resize_tensor_input(process_kp_driving_kp_driving_index, kp_driving.shape)
    process_kp_driving_interpreter.allocate_tensors()
    process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_driving_index, kp_driving)
    process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_source_index, kp_source)
    process_kp_driving_interpreter.set_tensor(process_kp_driving_relative_index, relative)
    process_kp_driving_interpreter.set_tensor(process_kp_driving_adapt_movement_scale_index, adapt_movement_scale)
    process_kp_driving_interpreter.invoke()
    return process_kp_driving_interpreter.get_tensor(process_kp_driving_output_index)


predictions = animate(source_image, frames, generator, kp_detector, process_kp_driving, 4, parser.relative, parser.adapt_movement_scale)
imageio.mimsave(parser.output + ".tflite.mp4", [img_as_ubyte(frame) for frame in predictions], fps=fps)
