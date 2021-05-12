import math
import tensorflow as tf
from logger import Visualizer
from tqdm import tqdm
from contextlib import nullcontext
import numpy as np

@tf.function
def first_elem_reshape(x):
    return x[0][None]

@tf.function
def first_elem_tile_reshape(x, tile):
    return tf.tile(x[0][None], tile)

@tf.function
def tile(x, tile):
    return tf.tile(x, tile)
    


@tf.function
def convert(x):
    return tf.identity(x)

def animate(source_image, driving_video, generator, kp_detector, process_kp_driving, 
            use_relative_movement=True, use_relative_jacobian=True, adapt_movement_scale=True,
            batch_size=4, exact_batch=False, profile=False, visualizer_params=None):
    
    
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32), tf.TensorSpec(shape=(), dtype=tf.int32)])
    def slice_driving(x, i, j):
        return x[i:j]
    
    l = len(driving_video)
    source_image, driving_video = convert(source_image), convert(driving_video)
    if profile:
        tf.profiler.experimental.start("./log")

    if exact_batch:
        kp_source = {k:first_elem_reshape(v) for k,v in kp_detector(tile(source_image, (batch_size, 1, 1, 1))).items()}
        start, end = np.array(0), np.array(batch_size * (len(driving_video) // batch_size))
        driving_video = slice_driving(driving_video, start, end)
        kp_driving_initial = {k:first_elem_reshape(v) for k,v in kp_detector(first_elem_tile_reshape(driving_video, (batch_size, 1, 1, 1))).items()}
    else:
        kp_source = kp_detector(source_image)
        kp_driving_initial = kp_detector(first_elem_reshape(driving_video))
    estimate_jacobian = 'jacobian' in kp_source.keys()
    
    predictions = []
    visualizations = []
    
    for i in tqdm(range(math.ceil(l / batch_size))):
        if profile:
            context = tf.profiler.experimental.Trace('animate', step_num=i, _r=1)
        else:
            context = nullcontext()
        with context:
            start = i * batch_size
            end = (i + 1) * batch_size
            if exact_batch and l - end < batch_size:
                continue
            start, end = np.array(start), np.array(end)
            driving_video_tensor = slice_driving(driving_video, start, end)
            kp_driving = kp_detector(driving_video_tensor)
            if estimate_jacobian:
                kp_norm = kp_driving if not use_relative_movement else process_kp_driving(
                    kp_driving['value'], kp_driving['jacobian'], kp_driving_initial['value'], kp_driving_initial['jacobian'], kp_source['value'], kp_source['jacobian'],
                    use_relative_movement, use_relative_jacobian, adapt_movement_scale,
                )
                out = generator(source_image, kp_norm['value'], kp_norm['jacobian'], kp_source['value'], kp_source['jacobian'])
            else:
                kp_norm = kp_driving if not use_relative_movement else process_kp_driving(
                    kp_driving['value'], kp_driving_initial['value'], kp_source['value'], use_relative_movement, adapt_movement_scale
                )
                out = generator(source_image, kp_norm['value'], kp_norm['value'])
            try:
                predictions.append(out['prediction'].numpy())
            except:
                predictions.append(out['prediction'])
            if batch_size == 1 and visualizer_params is not None:
                out['kp_driving'] = {k:v[0] for k,v in kp_driving.items()}
                out['kp_source'] = kp_source
                out['kp_norm'] = {k:v[i] for k,v in kp_norm.items()}
                try:
                    del out['sparse_deformed']
                except:
                    pass
                visualization = Visualizer(**visualizer_params).visualize(source=source_image[0], driving=driving_video[i], out=out)
                visualizations.append(visualization)
    
    if profile:
        tf.profiler.experimental.stop()
        
    return np.concatenate(predictions, 0), np.concatenate(visualizations, 0) if len(visualizations) > 0 else None
