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
    src_out = kp_detector(source_image)
    if isinstance(src_out, tuple) or isinstance(src_out, list):
        estimate_jacobian = True
        kp_source, kp_source_jacobian = src_out
    else:
        kp_source = src_out
    
    # Step 2: get kp_driving in batches
    kp_driving = []
    kp_driving_jacobian = []
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        driving_video_tensor = tf.convert_to_tensor(driving_video[start:end])
        if estimate_jacobian:
            kp_driving_frame_kp, kp_driving_frame_jacobian = kp_detector(driving_video_tensor)
            kp_driving_jacobian.append(kp_driving_frame_jacobian)
        else:
            kp_driving_frame_kp = kp_detector(driving_video_tensor)
        kp_driving.append(kp_driving_frame_kp)
    
    kp_driving = tf.concat(kp_driving, 0)
    if estimate_jacobian:
        kp_driving_jacobian = tf.concat(kp_driving_jacobian, 0)
    del driving_video
    
    # Step 3: process kp_driving
    kp_driving, kp_driving_jacobian = process_kp_driving(
        kp_driving, kp_driving_jacobian, kp_driving[0], kp_driving_jacobian[0], kp_source, kp_source_jacobian, use_relative_movement, use_relative_jacobian, adapt_movement_scale
    )
    
    # Step 4: get predictions in batches
    predictions = []
    for i in tqdm(range(math.ceil(l / batch_size))):
        start = i * batch_size
        end = (i + 1) * batch_size
        kp_driving_tensor = kp_driving[start:end]
        if estimate_jacobian:
            kp_driving_jacobian_tensor = kp_driving_jacobian[start:end]
            out = generator([source_image, kp_driving_tensor, kp_driving_jacobian_tensor, kp_source, kp_source_jacobian])
        else:
            out = generator([source_image, kp_driving_tensor, kp_source])
        predictions.append(out)
    
    if profile:
        tf.profiler.experimental.stop()
        
    return tf.concat(predictions, 0).numpy()
