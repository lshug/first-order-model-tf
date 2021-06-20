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
    if frames != -1:
        driving_video = driving_video[0 : frames]
    driving_video = [resize(frame, (frame_shape[0], frame_shape[1]))[..., :num_channels] for frame in driving_video]
    frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]
    return source_image, frames, fps

def save_video(path, predictions, fps):
    return imageio.mimsave(path, [img_as_ubyte(frame) for frame in predictions], fps=fps)

def save_visualization(path, visualizations):
    return imageio.mimsave(path, visualizations)

def save_frames_png(path, predictions):
    prediction = np.concatenate(predictions, 1)
    prediction = (255 * prediction).astype(np.uint8)
    imageio.imwrite(path, prediction)

def load_models_direct(model, prediction_only=False, static_batch_size=None, hardcode=None, prescale=False):
    config_path = f"config/{model}-256.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    frame_shape = config["dataset_params"]["frame_shape"]
    kp_detector = build_kp_detector(f"./checkpoint/{model}-cpk.pth.tar", **config["dataset_params"], **config["model_params"]["kp_detector_params"], **config["model_params"]["common_params"], 
                                    static_batch_size=static_batch_size, prescale=prescale)
    generator = build_generator(f"./checkpoint/{model}-cpk.pth.tar", not prediction_only, **config["dataset_params"], **config["model_params"]["generator_params"], 
                                     **config["model_params"]["common_params"], single_jacobian_map=kp_detector.single_jacobian_map, static_batch_size=static_batch_size, prescale=prescale)
    process_kp_driving = build_process_kp_driving(**config["model_params"]["common_params"], single_jacobian_map=kp_detector.single_jacobian_map, static_batch_size=static_batch_size, hardcode=hardcode)
    kp_detector, process_kp_driving, generator = tf.function(kp_detector), tf.function(process_kp_driving), tf.function(generator)    
    return kp_detector, process_kp_driving, generator, None

def load_models_savedmodel(model, prescale=False, **kwargs):
    kp_detector_loader = tf.saved_model.load("saved_models/" + model + "/kp_detector")
    kp_detector_base = kp_detector_loader.signatures["serving_default"]
    generator_loader = tf.saved_model.load("saved_models/" + model + "/generator")
    generator_base = generator_loader.signatures["serving_default"]
    generator_outs = list(generator_base.structured_outputs.keys())
    process_kp_driving_loader = tf.saved_model.load("saved_models/" + model + "/process_kp_driving")
    process_kp_driving_base = process_kp_driving_loader.signatures["serving_default"]
    estimate_jacobian = 'jacobian' in str(generator_base.structured_input_signature)
    kp_detector = lambda l: kp_detector_base(img=l)
    if prescale:
        generator = lambda *args: generator_base(source_image=args[0], kp_driving=args[1], kp_driving_jacobian=args[2], kp_source=args[3], kp_source_jacobian=args[4], source_image_scaled=args[5])
    else:
        generator = lambda *args: generator_base(source_image=args[0], kp_driving=args[1], kp_driving_jacobian=args[2], kp_source=args[3], kp_source_jacobian=args[4])
    if estimate_jacobian:
        process_kp_driving = lambda l, m, n, o, p, q, r, s: process_kp_driving_base(
                kp_driving=l,
                kp_driving_jacobian=m,
                kp_driving_initial=n,
                kp_driving_initial_jacobian=o,
                kp_source=p,
                kp_source_jacobian=q,
                use_relative_jacobian=tf.convert_to_tensor(r),
                adapt_movement_scale=tf.convert_to_tensor(s),
            )
    else:
        if prescale:
            generator = lambda l: generator_base(source_image=l[0], kp_driving=l[1], kp_source=l[2], source_image_scaled=l[3])
        else:
            generator = lambda l: generator_base(source_image=l[0], kp_driving=l[1], kp_source=l[2])
        process_kp_driving = lambda l, m, n, o: process_kp_driving_base(
                    kp_driving=l,
                    kp_driving_initial=m,
                    kp_source=n,                    
                    adapt_movement_scale=tf.convert_to_tensor(o),
                )
    kp_detector, process_kp_driving, generator = tf.function(kp_detector), tf.function(process_kp_driving), tf.function(generator)
    return kp_detector, process_kp_driving, generator, [[kp_detector_loader, kp_detector_base], [generator_loader, generator_base], [process_kp_driving_loader, process_kp_driving_base]]

def load_models_tflite(model, prescale=False, **kwargs):
    kp_detector_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/kp_detector.tflite")
    generator_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/generator.tflite")
    process_kp_driving_interpreter = tf.lite.Interpreter(model_path="tflite/" + model + "/process_kp_driving.tflite")
    kp_detector_base = kp_detector_interpreter.get_signature_runner()
    generator_base = generator_interpreter.get_signature_runner()
    process_kp_driving_base = process_kp_driving_interpreter.get_signature_runner()
    estimate_jacobian = any(['jacobian' in x['name'] for x in generator_interpreter.get_input_details()])
    kp_detector = lambda img: kp_detector_base(img=img)
    if estimate_jacobian:
        if prescale:
            generator = lambda *l: generator_base(source_image=l[0], kp_driving=l[1], kp_driving_jacobian=l[2], kp_source=l[3], kp_source_jacobian=l[4], source_image_scaled=l[5])
        else:
            generator = lambda *l: generator_base(source_image=l[0], kp_driving=l[1], kp_driving_jacobian=l[2], kp_source=l[3], kp_source_jacobian=l[4])        
        process_kp_driving = lambda l, m, n, o, p, q, r, s: process_kp_driving_base(
                kp_driving=l,
                kp_driving_jacobian=m,
                kp_driving_initial=n,
                kp_driving_initial_jacobian=o,
                kp_source=p,
                kp_source_jacobian=q,
                use_relative_jacobian=tf.convert_to_tensor(r),
                adapt_movement_scale=tf.convert_to_tensor(s),
                )
    else:
        generator = lambda *l: generator_base(source_image=l[0], kp_driving=l[1], kp_source=l[2], source_image_scaled=l[3])
        process_kp_driving = lambda l, m, n, o: process_kp_driving_base(
                    kp_driving=l,
                    kp_driving_initial=m,
                    kp_source=n,
                    adapt_movement_scale=tf.convert_to_tensor(o),
                    )
    return kp_detector, process_kp_driving, generator, [kp_detector_interpreter, process_kp_driving_interpreter, generator_interpreter]

def tflite_ops(model='vox'):
    ops = {}
    for module in ['kp_detector', 'generator', 'process_kp_driving']:
        ops[module] = list(set([x['op_name'] for x in tf.lite.Interpreter(model_path=f"tflite/{model}/" + module + '.tflite')._get_ops_details()]))
    return ops
