import tensorflow as tf
from firstordermodel import build_kp_detector, build_generator, build_process_kp_driving
import yaml
import argparse
import os
import subprocess
from tqdm import tqdm
import json

js_command_base = "tensorflowjs_converter --control_flow_v2=True --input_format=tf_saved_model --metadata= --saved_model_tags=serve --signature_name=serving_default --strip_debug_ops=True --weight_shard_size_bytes=4194304 saved_models/{0}/{1} js/{0}/{1}"
    

def build(checkpoint_path, config_path, output_name, module, prediction_only, hardcode, tfjs, static_batch_size, nolite, float16, prescale):
    js_command = js_command_base
    if float:
        js_command = js_command_base.replace('--metadata= ', '--metadata= --quantize_float16=* ')
        
    if not os.path.isdir("tflite/" + output_name):
        os.mkdir("tflite/" + output_name)

    if not os.path.isdir("saved_models/" + output_name):
        os.mkdir("saved_models/" + output_name)

    if tfjs and not os.path.isdir("js/" + output_name):
        os.mkdir("js/" + output_name)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    try:
        single_jacobian_map = config["model_params"]["kp_detector_params"]['single_jacobian_map']
    except:
        single_jacobian_map = False
    
    if module == 'kp_detector' or module=='all':
        kp_detector = build_kp_detector(checkpoint_path, **config["dataset_params"], **config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"], 
                                        static_batch_size=static_batch_size, prescale=prescale)
        print(f"{output_name} - kp_detector")
        tf.saved_model.save(kp_detector, "saved_models/" + output_name + "/kp_detector", kp_detector.__call__.get_concrete_function())
        if not nolite:
            kp_detector_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/kp_detector")
            if float16:
                kp_detector_converter.optimizations = [tf.lite.Optimize.DEFAULT]
                kp_detector_converter.target_spec.supported_types = [tf.float16]
            kp_detector_tflite = kp_detector_converter.convert()
            open("tflite/" + output_name + "/kp_detector.tflite", "wb").write(kp_detector_tflite)
            signature = tf.lite.Interpreter(model_content=kp_detector_tflite).get_signature_runner()
            tensor_index_map = {'inputs':dict(signature._inputs), 'outputs':dict(signature._outputs)}
            json.dump(tensor_index_map, open("tflite/" + output_name + "/kp_detector.json", 'w'))            
        if tfjs:
            command = js_command.format(output_name, 'kp_detector')
            subprocess.run(command.split())
    
    if module == 'generator' or module=='all':
        generator = build_generator(checkpoint_path, not prediction_only, **config["dataset_params"], **config["model_params"]["generator_params"], **config["model_params"]["common_params"], single_jacobian_map=single_jacobian_map, static_batch_size=static_batch_size, prescale=prescale)
        print(f"{output_name} - generator")
        tf.saved_model.save(generator, "saved_models/" + output_name + "/generator", signatures=generator.__call__.get_concrete_function())
        if not nolite:
            generator_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/generator")
            if float16:
                generator_converter.optimizations = [tf.lite.Optimize.DEFAULT]
                generator_converter.target_spec.supported_types = [tf.float16]
            generator_tflite = generator_converter.convert()
            open("tflite/" + output_name + "/generator.tflite", "wb").write(generator_tflite)
            signature = tf.lite.Interpreter(model_content=generator_tflite).get_signature_runner()
            tensor_index_map = {'inputs':dict(signature._inputs), 'outputs':dict(signature._outputs)}
            json.dump(tensor_index_map, open("tflite/" + output_name + "/generator.json", 'w'))            
        if tfjs:
            command = js_command.format(output_name, 'generator')
            subprocess.run(command.split())

    if module == 'process_kp_driving' or module=='all':
        process_kp_driving = build_process_kp_driving(**config["model_params"]["common_params"], 
                                                      single_jacobian_map=single_jacobian_map, static_batch_size=static_batch_size, hardcode=hardcode)
        print(f"{output_name} - process_kp_driving")
        tf.saved_model.save(process_kp_driving, "saved_models/" + output_name + "/process_kp_driving", process_kp_driving.__call__.get_concrete_function())
        if not nolite:
            process_kp_driving_converter = tf.lite.TFLiteConverter.from_saved_model("saved_models/" + output_name + "/process_kp_driving")
            if float16:
                process_kp_driving_converter.optimizations = [tf.lite.Optimize.DEFAULT]
                process_kp_driving_converter.target_spec.supported_types = [tf.float16]
            process_kp_driving_tflite = process_kp_driving_converter.convert()
            open("tflite/" + output_name + "/process_kp_driving.tflite", "wb").write(process_kp_driving_tflite)
            signature = tf.lite.Interpreter(model_content=process_kp_driving_tflite).get_signature_runner()
            tensor_index_map = {'inputs':dict(signature._inputs), 'outputs':dict(signature._outputs)}
            json.dump(tensor_index_map, open("tflite/" + output_name + "/process_kp_driving.json", 'w'))
        if tfjs:
            command = js_command.format(output_name, 'process_kp_driving')
            subprocess.run(command.split())

parser = argparse.ArgumentParser(description="Build saved_model, tflite, and tf.js modules from checkpoints and configs.")
parser.add_argument("--model", action="store", default="vox", help="model config and checkpoint to load")
parser.add_argument("-a", action="store_true", help="build models for all config files")
parser.add_argument('--module', choices=['all', 'kp_detector', 'generator', 'process_kp_driving'], default='all', help="module to build")
parser.add_argument('--nolite', action='store_true', help="don't build tflite modules")
parser.add_argument('--predictiononly', action="store_true", help="build the generator so that it only outputs predictions")
parser.add_argument('--float16', action="store_true", help="quantize lite to float16")
parser.add_argument('--tfjs', action='store_true', help="build tf.js models, requires tensorflowjs_converter")
parser.add_argument('--staticbatchsize', action='store', type=int, default=None, help="optional static batch size to use")
parser.add_argument('--hardcode', default=None, choices=['00', '01', '10', '11'],
                    help="optionally hardcode values for use_relative_jacobian and adapt_movement_scale at build type")
parser.add_argument("--prescale", dest="prescale", action="store_true", help="Reuse the result of AntiAliasInterpolation2d performed in kp_detector in the dense motion network")
parser.add_argument('--loadwithtorch', action="store_true",
                    help="use torch to load checkpoints instead of trying to load tensor buffers manually (requires pytorch)")
parser = parser.parse_args()

if parser.loadwithtorch:
    import load_torch_checkpoint
    load_torch_checkpoint.mode = 'torch'

print("Building")
if not parser.a:
    print(parser.model)
    checkpoint_path = f"checkpoint/{parser.model}-cpk.pth.tar"
    config_path = f"config/{parser.model}-256.yaml"
    output_name = config_path.split("/")[-1].split("256")[0][:-1]
    build(checkpoint_path, config_path, output_name, parser.module, parser.predictiononly, parser.hardcode, parser.tfjs, parser.staticbatchsize, parser.nolite, parser.float16, parser.prescale)
else:
    configs = os.listdir("config/")
    checkpoints = ["checkpoint/" + x.split("256")[0] + "cpk.pth.tar" for x in configs]
    output_names = [x.split("/")[-1].split("256")[0][:-1] for x in configs]
    configs = ["config/" + x for x in configs]
    for i, config in enumerate(tqdm(configs)):
        build(checkpoints[i], config, output_names[i], parser.module, parser.predictiononly, parser.hardcode, parser.tfjs, parser.staticbatchsize, parser.nolite, parser.float16, parser.prescale)

print("Done.")
