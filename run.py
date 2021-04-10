import argparse
import os
import yaml
from tqdm import tqdm
from animate import animate
from utils import load_image_video_pair, save_video, load_models_direct, load_models_savedmodel, load_models_tflite, save_multiple_pngs
from frames_dataset import FramesDataset, DatasetRepeater, PairedDataset

parser = argparse.ArgumentParser(description="Run inference")
parser.add_argument('--target', choices=['direct', 'savedmodel', 'tflite'], default='direct',
                    help="model version to run (between running the model directly, running the model's saved_model, and running its converted tflite")
#parser.add_argument('--mode', choices=['animate', 'reconstruct',], default='animate', help="Run mode (animate, reconstruct, or train)")
parser.add_argument('--datamode', choices=['file', 'dataset'], default='file', help='Data input mode (CLI-given file or config-defined dataset)')
parser.add_argument("--model", action="store", type=str, default="vox", help="model name")
parser.add_argument("--source_image", action="store", type=str, default="example/source.png", help="source image path for file datamode")
parser.add_argument("--driving_video", action="store", type=str, default="example/driving.mp4", help="driving video path for file datamode")
parser.add_argument("--output", action="store", type=str, default="example/output", help="output file name")
parser.add_argument("--dontappend",  action="store_true", help="don't append format name and .mp4 to the output filename")
parser.add_argument("--relative", action="store_true", help="relative kp mode")
parser.add_argument("--adapt", dest="adapt_movement_scale", action="store_true", help="adapt movement to the proportion between the sizes of subjects in the input image and the driving video")
parser.add_argument("--frames", type=int, default=-1, help="number of frames to process")
parser.add_argument("--batchsize", dest="batch_size", type=int, default=4, help="batch size")
parser.add_argument("--profile", action="store_true", help="enable tensorboard profiling")
parser = parser.parse_args()

load_funcs = {'direct':load_models_direct, 'savedmodel':load_models_savedmodel, 'tflite':load_models_tflite}

config_path = f"config/{parser.model}-256.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.Loader)
frame_shape = config['dataset_params']['frame_shape']
num_channels = config['model_params']['common_params']['num_channels']

kp_detector, process_kp_driving, generator, _interpreter_obj_list = load_funcs[parser.target](parser.model) 
format_appends = {'direct':'', 'savedmodel':'.savedmodel', 'tflite':'.tflite'}

# if parser.mode == 'animate':
if parser.datamode == 'file':
    source_image, frames, fps = load_image_video_pair(parser.source_image, parser.driving_video, frames=parser.frames, frame_shape=frame_shape, num_channels=num_channels)
    predictions = animate(source_image, frames, generator, kp_detector, process_kp_driving, parser.batch_size, parser.relative, parser.relative, parser.adapt_movement_scale, parser.profile)
    output = parser.output
    if not parser.dontappend:
        output = output + format_appends[parser.target] + '.mp4'
    save_video(output, predictions, fps=fps)
else:
    if not os.path.exists('./log/' + parser.model):
        os.mkdir('./log/' + parser.model)
    outdir = './log/' + parser.model
    if not parser.dontappend:
        outdir = outdir + format_appends[parser.target]
    dataset = FramesDataset(**config['dataset_params'])
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=config['animate_params']['num_pairs'])
    for idx, pair in tqdm(enumerate(dataset)):
       source_image, frames = pair['source_video'][0][None], pair['driving_video']
       predictions = animate(source_image, frames, generator, kp_detector, process_kp_driving, parser.batch_size, profile=parser.profile, **config['animate_params']['normalization_params'])
       result_name = f'{idx}_{pair["source_name"]}_{pair["driving_name"]}'
       full_outdir = outdir + '/' + result_name
       if not os.path.exists(full_outdir):
           os.mkdir(full_outdir)
       save_multiple_pngs(full_outdir, predictions)
       

print("Done.")
