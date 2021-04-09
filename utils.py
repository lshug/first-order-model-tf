import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import yaml
import tensorflow as tf
from firstordermodel import build_kp_detector, build_generator, build_process_kp_driving

def load_image_video_pair(img_path, video_path, frames=-1, frame_shape=(256, 256, 3), num_channels=3):
    source_image = imageio.imread(img_path)
    source_image = source_image[..., :num_channels]
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()["fps"]
    reader.close()
    driving_video = imageio.mimread(video_path, memtest=False)
    source_image = resize(source_image, (frame_shape[0], frame_shape[1]))[..., :num_channels][None].astype("float32")
    source = source_image.astype(np.float32)
    if frames != -1:
        driving_video = driving_video[0 : frames]
    driving_video = [resize(frame, (frame_shape[0], frame_shape[1]))[..., :num_channels] for frame in driving_video]
    frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]
    return source_image, frames, fps

def save_video(path, predictions, fps):
    return imageio.mimsave(path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

def load_models_direct(model):
    config_path = f"config/{model}-256.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    frame_shape = config["dataset_params"]["frame_shape"]
    kp_detector = build_kp_detector(f"./checkpoint/{model}-cpk.pth.tar", **config["dataset_params"], **config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"])
    generator_base = build_generator(f"./checkpoint/{model}-cpk.pth.tar", **config["dataset_params"], **config["model_params"]["generator_params"], **config["model_params"]["common_params"])
    generator = lambda arr: generator_base(arr[0], arr[1], arr[2], arr[3], arr[4])
    process_kp_driving = build_process_kp_driving(**config["model_params"]["common_params"], **config["model_params"]["kp_detector_params"])
    return kp_detector, process_kp_driving, generator, None

def load_models_savedmodel(model):
    kp_detector_loader = tf.saved_model.load("saved_models/" + model + "/kp_detector")
    kp_detector_base = kp_detector_loader.signatures["serving_default"]
    def kp_detector(l):
        kp_detector_out = kp_detector_base(img=l)
        return kp_detector_out["output_0"], kp_detector_out["output_1"]
    generator_loader = tf.saved_model.load("saved_models/" + model + "/generator")
    generator_base = generator_loader.signatures["serving_default"]
    generator = lambda l: generator_base(source_image=l[0], kp_driving=l[1], kp_driving_jacobian=l[2], kp_source=l[3], kp_source_jacobian=l[4])["output_0"]
    process_kp_driving_loader = tf.saved_model.load("saved_models/" + model + "/process_kp_driving")
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
    return kp_detector, process_kp_driving, generator, [[kp_detector_loader, kp_detector_base], [generator_loader, generator_base], [process_kp_driving_loader, process_kp_driving_base]]

def load_models_tflite(model):
    kp_detector_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/kp_detector.tflite")
    kp_detector_input_index = kp_detector_interpreter.get_input_details()[0]["index"]
    kp_detector_output1_index = [x for x in kp_detector_interpreter.get_output_details() if x["shape"].size == 3][0]["index"]
    kp_detector_output2_index = [x for x in kp_detector_interpreter.get_output_details() if x["shape"].size == 4][0]["index"]
    def kp_detector(img):
        kp_detector_interpreter.resize_tensor_input(kp_detector_input_index, img.shape)
        kp_detector_interpreter.allocate_tensors()
        kp_detector_interpreter.set_tensor(kp_detector_input_index, img)
        kp_detector_interpreter.invoke()
        return kp_detector_interpreter.get_tensor(kp_detector_output1_index), kp_detector_interpreter.get_tensor(kp_detector_output2_index)

    generator_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/generator.tflite")
    source_image_index = [x for x in generator_interpreter.get_input_details() if "source_image" in x["name"]][0]["index"]
    generator_kp_driving_index = [x for x in generator_interpreter.get_input_details() if "kp_driving" in x["name"] and "jacobian" not in x["name"]][0]["index"]
    generator_kp_driving_jacobian_index = [x for x in generator_interpreter.get_input_details() if "kp_driving_jacobian" in x["name"]][0]["index"]
    generator_kp_source_index = [x for x in generator_interpreter.get_input_details() if "kp_source" in x["name"] and "jacobian" not in x["name"]][0]["index"]
    generator_kp_source_jacobian_index = [x for x in generator_interpreter.get_input_details() if "kp_source_jacobian" in x["name"]][0]["index"]
    generator_output_index = generator_interpreter.get_output_details()[0]["index"]
    def generator(inputs):
        generator_interpreter.resize_tensor_input(source_image_index, inputs[0].shape)
        generator_interpreter.resize_tensor_input(generator_kp_driving_index, inputs[1].shape)
        generator_interpreter.resize_tensor_input(generator_kp_driving_jacobian_index, inputs[2].shape)
        generator_interpreter.resize_tensor_input(generator_kp_source_index, inputs[3].shape)
        generator_interpreter.resize_tensor_input(generator_kp_source_jacobian_index, inputs[4].shape)
        generator_interpreter.allocate_tensors()
        generator_interpreter.set_tensor(source_image_index, inputs[0])
        generator_interpreter.set_tensor(generator_kp_driving_index, inputs[1])
        generator_interpreter.set_tensor(generator_kp_driving_jacobian_index, inputs[2])
        generator_interpreter.set_tensor(generator_kp_source_index, inputs[3])
        generator_interpreter.set_tensor(generator_kp_source_jacobian_index, inputs[4])
        generator_interpreter.invoke()
        return generator_interpreter.get_tensor(generator_output_index)

    process_kp_driving_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/process_kp_driving.tflite")
    process_kp_driving_kp_driving_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_driving" in x["name"] and "jacobian" not in x["name"] and 'initial' not in x['name']][0]["index"]
    process_kp_driving_kp_driving_jacobian_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_driving_jacobian" in x["name"]][0]["index"]
    process_kp_driving_kp_driving_initial_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_driving_initial" in x["name"] and "jacobian" not in x["name"]][0]["index"]
    process_kp_driving_kp_driving_initial_jacobian_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_driving_initial_jacobian" in x["name"]][0]["index"]
    process_kp_driving_kp_source_index = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_source" in x["name"] and "jacobian" not in x["name"]][0]["index"]
    process_kp_driving_kp_source_jacobianindex = [x for x in process_kp_driving_interpreter.get_input_details() if "kp_source_jacobian" in x["name"]][0]["index"]
    process_kp_driving_relative_index = [x for x in process_kp_driving_interpreter.get_input_details() if "use_relative_movement" in x["name"]][0]["index"]
    process_kp_driving_relative_jacobian_index = [x for x in process_kp_driving_interpreter.get_input_details() if "use_relative_jacobian" in x["name"]][0]["index"]
    process_kp_driving_adapt_movement_scale_index = [x for x in process_kp_driving_interpreter.get_input_details() if "adapt_movement_scale" in x["name"]][0]["index"]
    process_kp_driving_output1_index = [x for x in process_kp_driving_interpreter.get_output_details() if x["shape"].size == 3][0]["index"]
    process_kp_driving_output2_index = [x for x in process_kp_driving_interpreter.get_output_details() if x["shape"].size == 4][0]["index"]
    def process_kp_driving(kp_driving, kp_driving_jacobian, kp_driving_initial, kp_driving_initial_jacobian, kp_source, kp_source_jacobian, use_relative_movement, use_relative_jacobian, adapt_movement_scale):
        process_kp_driving_interpreter.resize_tensor_input(process_kp_driving_kp_driving_index, kp_driving.shape)
        process_kp_driving_interpreter.resize_tensor_input(process_kp_driving_kp_driving_jacobian_index, kp_driving_jacobian.shape)        
        process_kp_driving_interpreter.allocate_tensors()
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_driving_index, kp_driving)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_driving_jacobian_index, kp_driving_jacobian)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_driving_initial_index, kp_driving_initial)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_driving_initial_jacobian_index, kp_driving_initial_jacobian)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_source_index, kp_source)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_kp_source_jacobianindex, kp_source_jacobian)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_relative_index, use_relative_movement)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_relative_jacobian_index, use_relative_jacobian)
        process_kp_driving_interpreter.set_tensor(process_kp_driving_adapt_movement_scale_index, adapt_movement_scale)
        process_kp_driving_interpreter.invoke()
        return process_kp_driving_interpreter.get_tensor(process_kp_driving_output1_index), process_kp_driving_interpreter.get_tensor(process_kp_driving_output2_index)
    return kp_detector, process_kp_driving, generator, [kp_detector_interpreter, process_kp_driving_interpreter, generator_interpreter]
