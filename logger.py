import numpy as np
import imageio

from firstordermodel import Interpolate
import tensorflow as tf

import os
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        image = self.visualizer.visualize(inp['driving'], inp['source'], out)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow'):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)

    def draw_image_with_kp(self, image, kp_array):
        image = np.copy(image)
        spatial_size = np.array(image.shape[:2][::-1])[np.newaxis]
        kp_array = spatial_size * (kp_array + 1) / 2
        num_kp = kp_array.shape[0]
        for kp_ind, kp in enumerate(kp_array):
            rr, cc = disk((kp[1], kp[0]), self.kp_size, shape=image.shape[:2])
            inside = self.colormap(kp_ind / num_kp)
            image[rr, cc] = np.array(inside)[:3]
        return image

    def create_image_column_with_kp(self, images, kp):
        image_array = np.array([self.draw_image_with_kp(v, k) for v, k in zip(images, kp)])
        return self.create_image_column(image_array)

    def create_image_column(self, images):
        if self.draw_border:
            images = np.copy(images)
            images[:, :, [0, -1]] = (1, 1, 1)
            images[:, :, [0, -1]] = (1, 1, 1)
        return np.concatenate(list(images), axis=0)

    def create_image_grid(self, *args):
        out = []
        for arg in args:
            if type(arg) == tuple:
                out.append(self.create_image_column_with_kp(arg[0], arg[1]))
            else:
                out.append(self.create_image_column(arg))
        return np.concatenate(out, axis=1)[None]

    def visualize(self, driving, source, out):
        images = []

        # Source image with keypoints
        kp_source = out['kp_source']['value'].numpy()[0]
        images.append((source[None], kp_source[None]))

        # Equivariance visualization
        if 'transformed_frame' in out.keys():
            transformed = out['transformed_frame'].numpy()
            transformed_kp = out['transformed_kp']['value'].numpy()
            images.append((transformed, transformed_kp))

        # Driving image with keypoints
        kp_driving = out['kp_driving']['value'].numpy()
        images.append((driving[None], kp_driving[None]))

        # Deformed image
        if 'deformed' in out.keys():
            deformed = out['deformed'].numpy()
            images.append(deformed)

        # Result with and without keypoints
        prediction = out['prediction'].numpy()
        if 'kp_norm' in out.keys():
            kp_norm = out['kp_norm']['value'].numpy()
            images.append((prediction, kp_norm[None]))
        images.append(prediction)


        ## Occlusion map
        if 'occlusion_map' in out.keys():
            occlusion_map = tf.tile(out['occlusion_map'], (1, 1, 1, 3))
            scales = (source.shape[0] / occlusion_map.shape[1], source.shape[1] / occlusion_map.shape[2])
            occlusion_map = Interpolate(scales)(occlusion_map).numpy()
            images.append(occlusion_map)

        # Deformed images according to each individual transform
        if 'sparse_deformed' in out.keys():
            full_mask = []            
            for i in range(out['sparse_deformed'].shape[1]): # for each kp
                image = out['sparse_deformed'][:, i, :, :, :]
                scales = (source.shape[0] / image.shape[1], source.shape[1] / image.shape[2])
                image = Interpolate(scales)(image).numpy()[0]
                mask = tf.tile(out['mask'][:, :, :, i:i+1], (1, 1, 1, 3))
                scales = (source.shape[0] / mask.shape[1], source.shape[1] / mask.shape[2])
                mask = Interpolate(scales)(mask)[0].numpy()
                if i != 0:
                    color = np.array(self.colormap((i - 1) / (out['sparse_deformed'].shape[1] - 1)))[:3]
                else:
                    color = np.array((0, 0, 0))

                color = color.reshape((1, 1, 3))
                images.append(image[None])
                if i != 0:
                    images.append((mask * color)[None])
                else:
                    images.append(mask[None])

                full_mask.append((mask * color)[None])
            images.append(sum(full_mask))
        image = self.create_image_grid(*images)
        image = (255 * image).astype(np.uint8)
        return image
