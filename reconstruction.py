import math
import tensorflow as tf
from logger import Visualizer
from tqdm import tqdm

def reconstruction(driving_video, generator, kp_detector, profile=False, visualizer_params=None):
    l = len(driving_video)
    batch_size = 1
    source_image = tf.convert_to_tensor(driving_video[0], "float32")[None]
    
    if profile:
        tf.profiler.experimental.start("./log")

    kp_source = kp_detector(source_image)
    estimate_jacobian = 'jacobian' in kp_source.keys()
    
    kp_driving_outs = []
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        driving_video_tensor = tf.convert_to_tensor(driving_video[start:end])
        kp_driving_out = kp_detector(driving_video_tensor)
        kp_driving_outs.append(kp_driving_out)
    kp_driving = {k:tf.concat([out[k] for out in kp_driving_outs], 0) for k in kp_source.keys()}
    kp_norm = kp_driving
    
    if visualizer_params is None:
        del driving_video
    
    predictions = []
    visualizations = []
    loss_list = []
    
    
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        kp_driving_batch = {k:v[start:end] for k,v in kp_norm.items()}
        if estimate_jacobian:
            out = generator([source_image, kp_driving_batch['value'], kp_driving_batch['jacobian'], kp_source['value'], kp_source['jacobian']])
        else:
            out = generator([source_image, kp_driving_batch['value'], kp_source['value']])
        predictions.append(out['prediction'])
        if batch_size == 1 and visualizer_params is not None:
            out['kp_driving'] = {k:v[0] for k,v in kp_driving_batch.items()}
            out['kp_source'] = kp_source
            try:
                del out['sparse_deformed']
            except:
                pass
            visualization = Visualizer(**visualizer_params).visualize(source=source_image[0], driving=driving_video[i], out=out)
            visualizations.append(visualization)
        loss_list.append(tf.reduce_mean(tf.abs(out['prediction'] - driving_video[i])))
    
    if profile:
        tf.profiler.experimental.stop()
        
    return tf.concat(predictions, 0).numpy(), tf.concat(visualizations, 0).numpy(), tf.reduce_mean(loss_list).numpy()
