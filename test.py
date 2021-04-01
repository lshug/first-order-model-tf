import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import tensorflow as tf
import numpy as np
from animate import animate 
import argparse


parser = argparse.ArgumentParser(description='Test inference using a saved_model')
parser.add_argument('--model', action='store', type=str, default='vox', nargs=1, dest='model', help='model name')
parser.add_argument('--source_image', action='store', dest='source_image', type=str, nargs=1, default='example/source.png', help='source image path')
parser.add_argument('--driving_video', action='store', dest='driving_video', type=str, nargs=1, default='example/driving.mp4', help='driving video path')
parser.add_argument('--output', action='store', type=str, default='output', nargs=1, dest='output', help='model name')
parser.add_argument('--relative',dest='relative', action='store_true')
parser.add_argument('--adapt',dest='adapt_movement_scale', action='store_true')
parser = parser.parse_args()


source_image = imageio.imread(parser.source_image)
source_image = source_image[..., :3]
reader = imageio.get_reader(parser.driving_video)
fps = reader.get_meta_data()['fps']
reader.close()
driving_video = imageio.mimread(parser.driving_video, memtest=False)
source_image = resize(source_image, (256, 256))[..., :3][None].astype('float32')
source = source_image.astype(np.float32)
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video][0:len(driving_video) if len(tf.config.list_physical_devices('GPU'))>0 else 200]
frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]

kp_detector_loader = tf.saved_model.load('saved_models/'+parser.model+'/kp_detector')
kp_detector_base = kp_detector_loader.signatures['serving_default']
kp_detector = lambda l: kp_detector_base(img=l)['output_0']

generator_loader = tf.saved_model.load('saved_models/'+parser.model+'/generator')
generator_base = generator_loader.signatures['serving_default']
generator = lambda l: generator_base(source_image=l[0],kp_driving=l[1],kp_source=l[2])['output_0']

process_kp_driving_loader = tf.saved_model.load('saved_models/'+parser.model+'/process_kp_driving')
process_kp_driving_base = process_kp_driving_loader.signatures['serving_default']
process_kp_driving = lambda l,m,n,o: process_kp_driving_base(kp_driving=l, kp_source=m, relative=tf.constant(n), adapt_movement_scale=tf.constant(o))['output_0']

predictions = animate(source_image,frames,generator,kp_detector,process_kp_driving,4,parser.relative,parser.adapt_movement_scale)
imageio.mimsave(parser.output+'.saved_model.mp4',[img_as_ubyte(frame) for frame in predictions], fps=fps)
