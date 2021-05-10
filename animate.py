import math
import tensorflow as tf
from logger import Visualizer
from tqdm import tqdm
from contextlib import nullcontext
import numpy as np

def animate(source_image, driving_video, generator, kp_detector, process_kp_driving, 
            use_relative_movement=True, use_relative_jacobian=True, adapt_movement_scale=True,
            batch_size=4, exact_batch=False, profile=False, visualizer_params=None):
    
    l = len(driving_video)
    
    if profile:
        tf.profiler.experimental.start("./log")

    if exact_batch:
        kp_source = {k:v[0][None] for k,v in kp_detector(np.tile(source_image, (batch_size, 1, 1, 1))).items()}
        driving_video = driving_video[:batch_size * (len(driving_video) // batch_size)]
        kp_driving_initial = {k:v[0][None] for k,v in kp_detector(np.tile(driving_video[0][None], (batch_size, 1, 1, 1))).items()}
    else:
        kp_source = kp_detector(source_image)
        kp_driving_initial = kp_detector(driving_video[0][None])
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
            driving_video_tensor = driving_video[start:end]
            kp_driving = kp_detector(driving_video_tensor)
            if estimate_jacobian:
                kp_norm = process_kp_driving(
                    kp_driving['value'], kp_driving['jacobian'], kp_driving_initial['value'], kp_driving_initial['jacobian'], kp_source['value'], kp_source['jacobian'],
                    use_relative_movement, use_relative_jacobian, adapt_movement_scale,
                )
                out = generator([source_image, kp_norm['value'], kp_norm['jacobian'], kp_source['value'], kp_source['jacobian']])
            else:
                kp_norm = process_kp_driving(
                    kp_driving['value'], kp_driving_initial['value'], kp_source['value'], use_relative_movement, adapt_movement_scale
                )
                out = generator([source_image, kp_norm['value'], kp_norm['value']])        
            predictions.append(out['prediction'].numpy())
            if batch_size == 1 and visualizer_params is not None:
                out['kp_driving'] = {k:v[0] for k,v in kp_driving.items()}
                out['kp_source'] = kp_source
                out['kp_norm'] = {k:v[i] for k,v in kp_norm.items()}
                try:
                    del out['sparse_deformed']
                except:
                    pass
                visualization = Visualizer(**visualizer_params).visualize(source=source_image[0], driving=driving_video[i].numpy(), out=out)
                visualizations.append(visualization)
    
    if profile:
        tf.profiler.experimental.stop()
        
    return np.concatenate(predictions, 0), tf.concat(visualizations, 0).numpy() if len(visualizations) > 0 else None
