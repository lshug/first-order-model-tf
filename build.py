import tensorflow as tf
from firstordermodel import build_kp_detector, build_generator, build_process_kp_driving
import yaml
import argparse
import os
from tqdm import tqdm


def build(checkpoint_path, config_path, output_name):
    if not os.path.isdir("tflite/" + output_name):
        os.mkdir("tflite/" + output_name)

    if not os.path.isdir("saved_models/" + output_name):
        os.mkdir("saved_models/" + output_name)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    kp_detector = build_kp_detector(checkpoint_path, **config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
    generator = build_generator(checkpoint_path, **config["model_params"]["generator_params"], **config["model_params"]["common_params"])
    process_kp_driving = build_process_kp_driving(**config["model_params"]["common_params"])

    print(f"{output_name} - kp_detector")
    tf.saved_model.save(kp_detector, "saved_models/" + output_name + "/kp_detector", kp_detector.__call__.get_concrete_function())
    kp_detector_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/kp_detector")
    kp_detector_converter.experimental_new_converter = True
    kp_detector_tflite = kp_detector_converter.convert()
    open("tflite/" + output_name + "/kp_detector.tflite", "wb").write(kp_detector_tflite)

    print(f"{output_name} - generator")
    tf.saved_model.save(generator, "saved_models/" + output_name + "/generator", signatures=generator.__call__.get_concrete_function())
    generator_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/generator")
    generator_converter.experimental_new_converter = True
    generator_tflite = generator_converter.convert()
    open("tflite/" + output_name + "/generator.tflite", "wb").write(generator_tflite)

    print(f"{output_name} - process_kp_driving")
    tf.saved_model.save(process_kp_driving, "saved_models/" + output_name + "/process_kp_driving", process_kp_driving.__call__.get_concrete_function())
    process_kp_driving_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/process_kp_driving")
    process_kp_driving_converter.experimental_new_converter = True
    process_kp_driving_tflite = process_kp_driving_converter.convert()
    open("tflite/" + output_name + "/process_kp_driving.tflite", "wb").write(process_kp_driving_tflite)


parser = argparse.ArgumentParser(description="Build saved_models and tflites from checkpoints and configs.")
parser.add_argument("--checkpoint_path", action="store", type=str, default="checkpoint/vox-cpk.pth.tar", nargs=1, dest="checkpoint_path", help="checkpoint path")
parser.add_argument("--config_path", action="store", dest="config_path", type=str, nargs=1, default="config/vox-256.yaml", help="config yaml path")
parser.add_argument("--a", dest="a", action="store_true")
parser = parser.parse_args()

print("Building")
if not parser.a:
    checkpoint_path = parser.checkpoint_path
    config_path = parser.config_path
    output_name = config_path.split("/")[-1].split("256")[0][:-1]
    build(checkpoint_path, config_path, output_name)
else:
    configs = os.listdir("config/")
    checkpoints = ["checkpoint/" + x.split("256")[0] + "cpk.pth.tar" for x in configs]
    output_names = [x.split("/")[-1].split("256")[0][:-1] for x in configs]
    configs = ["config/" + x for x in configs]
    for i, config in enumerate(tqdm(configs)):
        build(checkpoints[i], config, output_names[i])

print("Done.")
