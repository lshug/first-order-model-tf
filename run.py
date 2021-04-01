import yaml
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm
from animate import animate

parser = argparse.ArgumentParser(description='Build saved_models and tflites from checkpoints and configs.')
parser.add_argument('--model', action='store', type=str, default='vox', nargs=1, dest='model', help='model name')
parser.add_argument('--source_image', action='store', dest='source_image', type=str, nargs=1, default='source.png', help='source image path')
parser.add_argument('--driving_video', action='store', dest='driving_video', type=str, nargs=1, default='driving.mp4', help='driving video path')
parser.add_argument('--output', action='store', type=str, default='output', nargs=1, dest='output', help='model name')
parser.add_argument('--relative',dest='relative', action='store_true')
parser.add_argument('--adapt',dest='adapt_movement_scale', action='store_true')
parser.add_argument('--frames', dest='frames', type=int, default=0)
parser = parser.parse_args()

config_path = f'config/{parser.model}-256.yaml'
with open(config_path) as f:
    config = yaml.load(f)
num_channels = config['model_params']['common_params']['num_channels']
num_kp = config['model_params']['common_params']['num_kp']
kp_detector = build_kp_detector(f'./checkpoint/{parser.model}-cpk.pth.tar',**config['model_params']['kp_detector_params'],**config['model_params']['common_params'])
generator = build_generator(f'./checkpoint/{parser.model}-cpk.pth.tar', **config['model_params']['generator_params'],**config['model_params']['common_params'])
process_kp_driving = build_process_kp_driving(config['model_params']['common_params']['num_kp'])

source_image = imageio.imread(parser.source_image)
source_image = source_image[..., :num_channels]
reader = imageio.get_reader(parser.driving_video)
fps = reader.get_meta_data()['fps']
reader.close()
driving_video = imageio.mimread(parser.driving_video, memtest=False)
source_image = resize(source_image, (256, 256))[..., :num_channels][None].astype('float32')
source = source_image.astype(np.float32)
driving_video = [resize(frame, (256, 256))[..., :num_channels] for frame in driving_video][0:len(driving_video) if parser.frames==0 else parser.frames]
frames = np.array(driving_video)[np.newaxis].astype(np.float32)[0]


predictions = animate(source_image, frames,generator, kp_detector, process_kp_driving, 4, False, False)
imageio.mimsave(parser.output, [img_as_ubyte(frame) for frame in predictions], fps=fps)
print('Done.') 
