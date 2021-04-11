import math
import tensorflow as tf
from tqdm import tqdm

# General inference scheme:
# Step 1: get kp_source
# Step 2: get kp_driving in batches
# Step 3: process kp_driving
# Step 4: get predictions in batches
def animate(source_image, driving_video, generator, kp_detector, process_kp_driving, batch_size=4, use_relative_movement=True, use_relative_jacobian=True, adapt_movement_scale=True, profile=False):
    l = len(driving_video)
    source_image = tf.convert_to_tensor(source_image, "float32")
    
    
    if profile:
        tf.profiler.experimental.start("./log")

    # Step 1: get kp_source
    kp_source = kp_detector(source_image)
    estimate_jacobian = 'jacobian' in kp_source.keys()
    
    # Step 2: get kp_driving in batches
    kp_driving_outs = []
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        driving_video_tensor = tf.convert_to_tensor(driving_video[start:end])
        kp_driving_out = kp_detector(driving_video_tensor)
        kp_driving_outs.append(kp_driving_out)
    kp_driving = {k:tf.concat([out[k] for out in kp_driving_outs], 0) for k in kp_source.keys()}
    del driving_video
    
    # Step 3: process kp_driving
    if estimate_jacobian:
        kp_norm = process_kp_driving(
            kp_driving['value'], kp_driving['jacobian'], kp_driving['value'][0], kp_driving['jacobian'][0], kp_source['value'], kp_source['jacobian'],
            use_relative_movement, use_relative_jacobian, adapt_movement_scale,
        )
    else:
        kp_norm = process_kp_driving(
            kp_driving['value'], kp_driving['value'][0], kp_source['value'], use_relative_movement, adapt_movement_scale
        )
    
    # Step 4: get predictions in batches
    predictions = []
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        kp_driving_batch = {k:v[start:end] for k,v in kp_norm.items()}
        if estimate_jacobian:
            out = generator([source_image, kp_driving_batch['value'], kp_driving_batch['jacobian'], kp_source['value'], kp_source['jacobian']])
        else:
            out = generator([source_image, kp_driving_batch['value'], kp_source['value']])
        predictions.append(out['prediction'])
    
    if profile:
        tf.profiler.experimental.stop()
        
    return tf.concat(predictions, 0).numpy()
