import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from load_torch_checkpoint import load_torch_checkpoint

class AntiAliasInterpolation2d(layers.Layer):
    def __init__(self, channels, scale, static_batch_size=None, **kwargs):
        sigma = 1.5 #(1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ka = ka
        self.kb = kb
        self.groups = channels
        self.scale = scale
        self.static_batch_size = static_batch_size

        super(AntiAliasInterpolation2d, self).__init__(**kwargs)

    def build(self, input_shape):
        self.inpshape = input_shape
        kernel_size = self.kernel_size
        sigma = self.sigma
        kernel = tf.ones(1, dtype=K.floatx())
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        meshgrids = tf.meshgrid(*(tf.cast(tf.range(n), K.floatx()) for n in kernel_size))
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * tf.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel, (1, 1, *kernel.shape))
        kernel = tf.tile(kernel, (self.groups, 1, 1, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        self.kernel = self.add_weight("kernel", kernel.shape, trainable=False)
        self.set_weights([kernel])
        self.interpolate = Interpolate((self.scale, self.scale), static_batch_size=self.static_batch_size)
        ks = []
        for i in range(self.groups):
            ks.append(self.kernel_slice(self.kernel, i))
        self.ks = ks
        super(AntiAliasInterpolation2d, self).build(input_shape)

    def kernel_slice(self, x, i):
        return tf.expand_dims(x[:, :, :, i], 3)

    def call(self, x):
        out = layers.ZeroPadding2D((self.ka, self.kb))(x)
        outputs = [0] * self.groups
        for i in range(self.groups):
            k = self.ks[i]
            im = tf.reshape(out[:,:,:,i:i+1], (-1, out.shape[1], out.shape[2], 1))
            outputs[i] = tf.nn.conv2d(im, k, strides=(1, 1, 1, 1), padding="VALID")
        out = tf.concat(outputs, 3)
        out = self.interpolate(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.scale, input_shape[2] * self.scale, input_shape[3])

    def get_config(self):
        config = super(AntiAliasInterpolation2d, self).get_config()
        config.update({"kernel_size": self.kernel_size, "groups": self.groups, "ka": self.ka, "kb": self.kb, "scale": self.scale, "static_batch_size":self.static_batch_size})
        return config


def make_coordinate_grid(spatial_size, dtype):
    h, w = spatial_size
    h = tf.cast(h, dtype)
    w = tf.cast(w, dtype)
    x = tf.cast(tf.range(w, dtype='float32'), dtype)
    y = tf.cast(tf.range(h, dtype='float32'), dtype)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    yy = tf.tile(tf.reshape(y, (-1, 1)), (1, w))
    xx = tf.tile(tf.reshape(y, (1, -1)), (h, 1))
    meshed = tf.concat([tf.expand_dims(xx, 2), tf.expand_dims(yy, 2)], 2)
    return meshed


class GaussianToKpTail(layers.Layer):
    def __init__(self, temperature=0.1, spatial_size=(58, 58), num_kp=10, **kwargs):
        self.temperature = temperature
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(GaussianToKpTail, self).__init__(**kwargs)

    def call(self, x):
        out = tf.reshape(x, (-1, self.spatial_size[0] * self.spatial_size[1], self.num_kp))  # B (H*W) 3
        out = keras.activations.softmax(out / self.temperature, axis=1)
        heatmap = tf.reshape(out, (-1, self.spatial_size[0], self.spatial_size[1], self.num_kp))
        heatmap2 = tf.transpose(heatmap, (0, 3, 1, 2))
        heatmap2 = tf.reshape(heatmap2, (-1, self.num_kp, self.spatial_size[0] * self.spatial_size[1], 1))
        heatmap2 = tf.tile(heatmap2, (1, 1, 1, 2))
        heatmap2 = tf.reshape(heatmap2, (-1, self.num_kp * self.spatial_size[0] * self.spatial_size[1], 2))
        mult = heatmap2 * self.grid
        mult = tf.reshape(mult, (-1, self.num_kp, self.spatial_size[0] * self.spatial_size[1], 2))
        value = tf.reduce_sum(mult, 2)
        return value, heatmap

    def build(self, input_shape):
        grid = make_coordinate_grid(self.spatial_size, K.floatx())[None][None]
        grid = tf.tile(grid, (1, self.num_kp, 1, 1, 1))
        grid = tf.reshape(grid, (-1, self.num_kp * self.spatial_size[0] * self.spatial_size[1], 2))
        self.grid = grid
        super(GaussianToKpTail, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3], 2), (input_shape[0], input_shape[1], input_shape[2], self.num_kp)

    def get_config(self):
        config = super(GaussianToKpTail, self).get_config()
        config.update({"temperature": self.temperature, "spatial_size": self.spatial_size, "num_kp": self.num_kp})
        return config


class SparseMotion(layers.Layer):
    def __init__(self, spatial_size, num_kp, estimate_jacobian, static_batch_size=None, **kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        self.estimate_jacobian = estimate_jacobian
        self.static_batch_size = static_batch_size
        super(SparseMotion, self).__init__(**kwargs)
    
    def batch_batch_four_by_four_inv(self, x):
        num_kp = self.num_kp
        a = x[:, :, 0:1, 0:1]
        b = x[:, :, 0:1, 1:2]
        c = x[:, :, 1:2, 0:1]
        d = x[:, :, 1:2, 1:2]
        dets = a * d - b * c
        dets = tf.reshape(dets, (-1, num_kp, 1, 1))
        d = tf.reshape(d, (-1, num_kp, 1))
        c = tf.reshape(c, (-1, num_kp, 1))
        b = tf.reshape(b, (-1, num_kp, 1))
        a = tf.reshape(a, (-1, num_kp, 1))
        row1 = tf.concat([d, (-1 * c)], 2)
        row2 = tf.concat([(-1 * b), a], 2)
        row1 = tf.reshape(row1, (-1, num_kp, 2, 1))
        row2 = tf.reshape(row2, (-1, num_kp, 2, 1))
        m = tf.concat([row1, row2], 3)        
        return m / dets

    def call(self, x):
        kp_driving, kp_source = x[0], x[2]
        if self.static_batch_size is None:
            bs = tf.shape(kp_driving)[0]
        else:
            bs = self.static_batch_size
        h, w = self.spatial_size
        identity_grid = self.grid  # h*w 2
        coordinate_grid = identity_grid - tf.reshape(kp_driving, (-1, self.num_kp, 1, 2)) # b kp h*w 2
        identity_grid = identity_grid + tf.zeros_like(tf.reduce_sum(coordinate_grid, 1))  # b h*w 2
        identity_grid = tf.reshape(identity_grid, (-1, 1, h * w, 2))

        # adjust coordinate grid with jacobians
        if self.estimate_jacobian:
            kp_driving_jacobian, kp_source_jacobian = x[1], x[3]
            jacobian_number = kp_driving_jacobian.shape[1]
            left = tf.reshape(kp_source_jacobian, (-1, 2, 2)) # 4*bs 2 2. untiled: 4 2 2
            right = self.batch_batch_four_by_four_inv(kp_driving_jacobian)
            if self.static_batch_size is None:
                jacobian = left @ right # b 10 2 2
                jacobian = tf.tile(jacobian, (1, self.jacobian_tile, 1, 1))
                reshaped_jacobian = tf.reshape(jacobian, (-1, 2, 2))
                reshaped_grid = tf.transpose(tf.reshape(coordinate_grid, (-1, h * w, 2, 1)), (1,0,2,3))
                coordinate_grid = tf.reshape(tf.transpose((reshaped_jacobian @ reshaped_grid),(1,0,2,3)), (-1, self.num_kp, h * w, 2)) # b*kp h w 2
            else:
                right, left = tf.transpose(left, (0,2,1)), tf.transpose(right, (0,1,3,2))                
                res = []
                for i in range(jacobian_number):
                    lt = tf.reshape(left[:, i:i+1, :, :], (-1, 2))[:]
                    rt = tf.reshape(right[None][:, i:i+1], (2, 2))
                    out = tf.nn.bias_add(lt @ rt, tf.ones(2, dtype=K.floatx()) * 1e-20)
                    res.append(tf.reshape(out, (self.static_batch_size, 1, 2, 2))) # b 1 2 2
                jacobian_inter = tf.concat(res, 1)                
                jacobian = tf.transpose(jacobian_inter, (0,1,3,2))                
                jacobian = tf.tile(jacobian, (1, self.jacobian_tile, 1, 1))
                reshaped_jacobian = tf.reshape(jacobian, (-1, 2, 2))
                reshaped_grid = tf.reshape(coordinate_grid, (-1, h * w, 2))                
                out = []
                r = self.static_batch_size * self.num_kp
                for i in range(r):
                    right = tf.transpose(tf.reshape(reshaped_jacobian[None][:, i:i+1], (2, 2)))
                    left = tf.reshape(reshaped_grid[None][:, i:i+1], (h * w, 2))[:]
                    o = tf.nn.bias_add(left @ right, tf.ones(2, dtype=K.floatx()) * 1e-20)
                    out.append(tf.reshape(o, (1, h*w, 2)))
                coordinate_grid = tf.reshape(tf.concat(out, 0), (self.static_batch_size, self.num_kp, h * w, 2))                

        mult = tf.tile(tf.reshape(kp_source, (-1, self.num_kp, 1, 2)), (bs, 1, h * w, 1))
        mult = 0 - mult
        driving_to_source = coordinate_grid - mult # b kp h*w 2

        sparse_motions = tf.concat([identity_grid, driving_to_source], 1) # b kp+1 h*w 2
        return sparse_motions

    def build(self, input_shape):
        self.grid = tf.reshape(make_coordinate_grid(self.spatial_size, K.floatx()), (-1, 2))
        if self.estimate_jacobian:
            self.jacobian_tile = self.num_kp if input_shape[1][1]==1 else 1
        super(SparseMotion, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, self.num_kp + 1, self.spatial_size[0], self.spatial_size[1], 2)

    def get_config(self):
        config = super(SparseMotion, self).get_config()
        config.update({"spatial_size":self.spatial_size, "num_kp":self.num_kp, "estimate_jacobian":self.estimate_jacobian, "static_batch_size":self.static_batch_size})
        return config


class Deform(layers.Layer):
    def __init__(self, spatial_size, channels, num_kp, static_batch_size=None, **kwargs):
        self.spatial_size = spatial_size
        self.channels = channels
        self.num_kp = num_kp
        self.static_batch_size = static_batch_size
        super(Deform, self).__init__(**kwargs)

    def call(self, x):
        if self.static_batch_size is None:
            bs = tf.shape(x[1])[0]
        else:
            bs = self.static_batch_size
        h, w = self.spatial_size
        source_repeat = tf.tile(x[0], (bs * (self.num_kp + 1), 1, 1, 1)) # (b*11, h, w, 3)
        sparse_motions = tf.reshape(x[1], (-1, h, w, 2))
        sparse_deformed = self.grid_sample([source_repeat, sparse_motions])
        sparse_deformed = tf.reshape(sparse_deformed, (-1, h * w, self.channels))
        return sparse_deformed

    def build(self, input_shape):
        if self.static_batch_size is None:
            self.grid_sample = GridSample()
        else:
            self.grid_sample = GridSample(static_batch_size=self.static_batch_size * (self.num_kp + 1))
        super(Deform, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][1], self.spatial_size[0], self.spatial_size[1], self.channels)

    def get_config(self):
        config = super(Deform, self).get_config()
        config.update({"spatial_size": self.spatial_size, "channels": self.channels, "num_kp": self.num_kp})
        return config


class KpToGaussian(layers.Layer):
    def __init__(self, spatial_size, num_kp, static_batch_size=None, **kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        self.static_batch_size = static_batch_size
        super(KpToGaussian, self).__init__(**kwargs)

    def call(self, x):
        kp_variance = 0.01
        mean = x
        grid = self.grid  # 1 1 H*W 2
        mean = tf.reshape(mean, (-1, self.num_kp, 1, 2))  # B 10 1 2
        mean_sub = grid - mean
        mean_sub = self.reshape(mean_sub)
        if self.static_batch_size is None:
            mean_sub = tf.square(mean_sub)
        else:
            mean_sub = mean_sub * mean_sub
        mean_sub = tf.reshape(mean_sub, (-1, self.num_kp, self.spatial_size[0] * self.spatial_size[1], 2))
        out = tf.math.exp(-0.5 * tf.reduce_sum(mean_sub, -1) / kp_variance)
        return out

    def build(self, input_shape):
        grid = tf.reshape(make_coordinate_grid(self.spatial_size, K.floatx()), (1, 1, self.spatial_size[0] * self.spatial_size[1], 2))
        grid = tf.tile(grid, (1, self.num_kp, 1, 1))
        self.grid = grid
        self.reshape = layers.Lambda(lambda l: tf.reshape(l, (-1, self.num_kp * self.spatial_size[0] * self.spatial_size[1], 2)), name='kptogaussianreshape')
        super(KpToGaussian, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_kp, self.spatial_size[0] * self.spatial_size[1])

    def get_config(self):
        config = super(KpToGaussian, self).get_config()
        config.update({"spatial_size": self.spatial_size, "num_kp": self.num_kp})
        return config


class Interpolate(layers.Layer):
    def __init__(self, scale_factor, static_batch_size=None, **kwargs):
        self.scale_factor = tuple(float(x) for x in scale_factor)
        self.static_batch_size = static_batch_size
        super(Interpolate, self).__init__(**kwargs)

    def build(self, input_shape):        
        H, W = input_shape[1], input_shape[2]
        H_f, W_f = tf.cast(H, K.floatx()), tf.cast(W, K.floatx())
        y_max = tf.floor(input_shape[1] * self.scale_factor[0])
        x_max = tf.floor(input_shape[2] * self.scale_factor[1])
        x_scale = tf.cast(input_shape[2] / x_max, K.floatx())
        y_scale = tf.cast(input_shape[1] / y_max, K.floatx())
        grid = tf.cast(tf.transpose(tf.stack(tf.meshgrid(tf.range(x_max), tf.range(y_max))[::-1]), (1, 2, 0)), K.floatx())
        grid_shape = (1, *grid.shape)
        flat_grid = tf.reshape(grid, (-1, 2))
        flat_grid = tf.stack(([(tf.math.minimum(tf.floor((flat_grid[..., 0]) * y_scale), H_f - 1)), tf.math.minimum(tf.floor((flat_grid[..., 1]) * x_scale), W_f - 1)]))
        flat_grid = tf.transpose(flat_grid, (1, 0))
        grid = tf.reshape(flat_grid, grid_shape)
        grid = tf.cast(grid, "int32")
        self.y_max = int((input_shape[1] * self.scale_factor[0]) // 1)
        self.x_max = int((input_shape[2] * self.scale_factor[1]) // 1)
        self.grid = tf.reshape(grid, (1, self.y_max, self.x_max, 2))
        if self.static_batch_size is not None:
            N = self.static_batch_size
            img_shape = (N, *tuple(input_shape[1:-1]))
            batch_range = tf.reshape(tf.range(self.static_batch_size), (-1, 1, 1, 1))
            g = tf.tile(tf.reshape(batch_range, (-1, 1, 1, 1)), (1, y_max, x_max, 1)) # batch_range, batch, y_max, x_max
            grid = tf.tile(grid, (N, 1, 1, 1))
            grid = tf.concat([g, grid], 3)
            
            img_shape = (N, *tuple(input_shape[1:-1]))
            self.inp_shape = (*input_shape,)
            self.final_shape = (*grid.shape[:-1], input_shape[-1])
            grid = tf.reshape(grid, (-1, 3))
            s = 0
            for i in range(3):
                coef = tf.reduce_prod(img_shape[i+1:]).numpy()
                s += coef * grid[:, i]                
            self.static_grid = s.numpy()
        super(Interpolate, self).build(input_shape)

    def call(self, img):
        if self.static_batch_size is None:
            grid = self.grid
            y_max = self.y_max
            x_max = self.x_max
            N = tf.shape(img)[0]
            batch_range = tf.reshape(tf.range(N), (-1, 1, 1, 1))
            g = tf.tile(tf.reshape(batch_range, (-1, 1, 1, 1)), (1, y_max, x_max, 1)) # batch_range, batch, y_max, x_max
            grid = tf.tile(grid, (N, 1, 1, 1))
            grid = tf.concat([g, grid], 3)
            out = tf.gather_nd(img, grid)
        else:
            out = tf.reshape(tf.gather(tf.reshape(img, (-1, self.inp_shape[3])), self.static_grid), self.final_shape)
        return out
    
    def compute_output_shape(self, input_shape):
        scale_factor = self.scale_factor
        return input_shape[0], input_shape[1] * scale_factor[0], input_shape[2] * scale_factor[1], input_shape[3]

    def get_config(self):
        config = super(Interpolate, self).get_config()
        config.update({"scale_factor":self.scale_factor, "static_batch_size":self.static_batch_size})
        return config


class BilinearInterpolate(layers.Layer):
    def __init__(self, size, static_batch_size=None, **kwargs):
        self.size = size
        self.static_batch_size = static_batch_size
        super(BilinearInterpolate, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.static_batch_size is not None:
            self.brange = tf.range(self.static_batch_size)
        size = self.size
        x_scale = tf.cast(input_shape[2] / size[1], K.floatx())
        y_scale = tf.cast(input_shape[1] / size[0], K.floatx())
        H, W = input_shape[1], input_shape[2]
        zeros = tf.zeros((size[0], size[1]), dtype="int32")
        ones = tf.expand_dims(tf.ones((size[0], size[1]), dtype="int32"), 2)
        self.zeros = zeros
        self.ones = ones

        grid = tf.cast(tf.transpose(tf.stack(tf.meshgrid(tf.range(size[1]), tf.range(size[0]))[::-1]), (1, 2, 0)), K.floatx())
        grid_shape = (size[1], size[0], 2)
        flat_grid = tf.reshape(grid, (-1, 2))
        flat_grid = tf.stack([(flat_grid[:, 0] + 0.5) * y_scale - 0.5, (flat_grid[:, 1] + 0.5) * x_scale - 0.5])
        flat_grid = tf.transpose(flat_grid, (1, 0))
        flat_grid = tf.math.maximum(flat_grid, 0)
        grid = tf.reshape(flat_grid, grid_shape)
        grid_int = tf.cast(tf.floor(grid), "int32")
        grid_float = tf.cast(grid_int, K.floatx())
        y = tf.cast(tf.transpose(tf.stack([tf.cast(grid_int[..., 0] < W - 1, "int32"), zeros]), (1, 2, 0)), "int32")
        x = tf.cast(tf.transpose(tf.stack([zeros, tf.cast(grid_int[..., 1] < H - 1, "int32")]), (1, 2, 0)), "int32")
        
        grid00 = grid_int
        grid01 = grid00 + x
        grid10 = grid00 + y
        grid11 = grid00 + y + x        
        
        if self.static_batch_size is not None:
            N = self.static_batch_size
            batch_range = tf.range(N)
            g = tf.cast(tf.expand_dims(tf.tile(tf.reshape(batch_range, (-1, 1, 1)), (1, size[0], size[1])), 3), "int32")
            grid00 = tf.concat([g, tf.tile(tf.reshape(grid00, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid01 = tf.concat([g, tf.tile(tf.reshape(grid01, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid10 = tf.concat([g, tf.tile(tf.reshape(grid10, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid11 = tf.concat([g, tf.tile(tf.reshape(grid11, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            
            img_shape = (N, *tuple(input_shape[1:-1]))
            self.inp_shape = (*input_shape,)
            self.final_shape = (*grid00.shape[:-1], input_shape[-1])
            grid00 = tf.reshape(grid00, (-1, 3))
            grid01 = tf.reshape(grid01, (-1, 3))
            grid10 = tf.reshape(grid10, (-1, 3))
            grid11 = tf.reshape(grid11, (-1, 3))
            s00, s01, s10, s11 = 0, 0, 0, 0
            for i in range(3):
                coef = tf.reduce_prod(img_shape[i+1:]).numpy()
                s00 += coef * grid00[:, i]
                s01 += coef * grid01[:, i]
                s10 += coef * grid10[:, i]
                s11 += coef * grid11[:, i]
            self.s00 = s00
            self.s01 = s01
            self.s10 = s10
            self.s11 = s11
                    
        self.grid00 = grid00.numpy()
        self.grid01 = grid01.numpy()
        self.grid10 = grid10.numpy()
        self.grid11 = grid11.numpy()
        self.grid = grid
        self.grid_int = grid_int
        self.grid_float = grid_float
        
        super(BilinearInterpolate, self).build(input_shape)

    def call(self, img):
        size = self.size
        
        H, W = img.shape[1], img.shape[2]
        zeros = self.zeros
        ones = self.ones
        grid = self.grid
        grid_float = self.grid_float
        
        grid00 = self.grid00
        grid01 = self.grid01
        grid10 = self.grid10
        grid11 = self.grid11
        
        if self.static_batch_size is None:
            N = tf.shape(img)[0]
            batch_range = tf.range(N)
            g = tf.cast(tf.expand_dims(tf.tile(tf.reshape(batch_range, (-1, 1, 1)), (1, size[0], size[1])), 3), "int32")
            grid00 = tf.concat([g, tf.tile(tf.reshape(grid00, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid01 = tf.concat([g, tf.tile(tf.reshape(grid01, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid10 = tf.concat([g, tf.tile(tf.reshape(grid10, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            grid11 = tf.concat([g, tf.tile(tf.reshape(grid11, (1, size[0], size[1], 2)), (N, 1, 1, 1))], 3)
            val00 = tf.gather_nd(img, grid00)
            val01 = tf.gather_nd(img, grid01)
            val10 = tf.gather_nd(img, grid10)
            val11 = tf.gather_nd(img, grid11)
        else:
            N = self.static_batch_size
            val00 = tf.reshape(tf.gather(tf.reshape(img, (-1, self.inp_shape[3])), self.s00), self.final_shape)
            val01 = tf.reshape(tf.gather(tf.reshape(img, (-1, self.inp_shape[3])), self.s01), self.final_shape)
            val10 = tf.reshape(tf.gather(tf.reshape(img, (-1, self.inp_shape[3])), self.s10), self.final_shape)
            val11 = tf.reshape(tf.gather(tf.reshape(img, (-1, self.inp_shape[3])), self.s11), self.final_shape)

        l = tf.tile(tf.reshape(grid, (1, size[0], size[1], 2)), (N, 1, 1, 1)) - grid_float

        w0 = tf.reshape(l[..., 1:2], (-1, size[0], size[1], 1)) #-1, H, W, 1
        w1 = 1 - w0
        h0 = tf.reshape(l[..., 0:1], (-1, size[0], size[1], 1))
        h1 = 1 - h0

        return h1 * (w1 * val00 + w0 * val01) + h0 * (w1 * val10 + w0 * val11)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.size[0], self.size[1], input_shape[3]

    def get_config(self):
        config = super(BilinearInterpolate, self).get_config()
        config.update({"size":self.size, "static_batch_size":self.static_batch_size})
        return config


class GridSample(layers.Layer):
    def __init__(self, static_batch_size=None, **kwargs):
        self.static_batch_size = static_batch_size
        super(GridSample, self).__init__(**kwargs)

    def call(self, data):
        return self._grid_sample(data)
    
    def grt(self, x, y):
        z = x - y
        z = tf.math.maximum(z, 0.0)
        z = z / (z + 1.0)
        z = -1.0 * (tf.floor(1.0 - z) - 1.0)
        return z
    
    def _pseudo_gather_nd(self, img, grid):
        img_shape, final_shape = self.img_shape, self.final_shape
        img = tf.reshape(img, (-1, img_shape[-1]))
        grid = tf.cast(tf.reshape(grid, (-1, len(img_shape) - 1)), 'int32')
        s = 0
        for i in range(len(img_shape) - 1):
            s += self.coefs[i] * tf.reshape(grid[None][:, :, i:i+1], (-1,))
        return tf.reshape(tf.gather(img, tf.cast(s, 'int32')), final_shape)

    def _grid_sample(self, data):
        img = data[0]
        grid = data[1]

        H, W = self.H, self.W
        iH, iW = self.iH, self.iW

        # extract x,y from grid
        x = tf.reshape(grid[:, :, :, 0:1], (-1, self.i_H, self.i_W, 1))
        y = tf.reshape(grid[:, :, :, 1:2], (-1, self.i_H, self.i_W, 1))

        # ComputeLocationBase
        x = (x + 1) * (W / 2) - 0.5
        y = (y + 1) * (H / 2) - 0.5

        # compute_interp_params
        x_w = tf.floor(x)
        y_n = tf.floor(y)
        x_e = x_w + 1
        y_s = y_n + 1

        w = x - x_w
        e = 1 - w
        n = y - y_n
        s = 1 - n

        nw = s * e
        ne = s * w
        sw = n * e
        se = n * w
                
        # forward
        w_mask = self.grt(x_w, -1.0) * self.grt(float(self.i_W), x_w)
        n_mask = self.grt(y_n, -1.0) * self.grt(float(self.i_H), y_n)
        e_mask = self.grt(x_e, -1.0) * self.grt(float(self.i_W), x_e)
        s_mask = self.grt(y_s, -1.0) * self.grt(float(self.i_H), y_s)
        
        nw_mask = w_mask * n_mask
        ne_mask = e_mask * n_mask
        sw_mask = s_mask * w_mask
        se_mask = e_mask * s_mask

        # loop
        if self.static_batch_size is None:
            N = tf.shape(x)[0]
            b = tf.cast(tf.range(N, dtype='float32'), K.floatx())
            b = tf.reshape(b, (-1, 1, 1, 1))
            b = tf.tile(b, (1, H, W, 1))         
        else:
            b = self.brange
        

        grid_nw = tf.concat([b, y_n, x_w], 3)
        grid_nw = nw_mask * grid_nw
        if self.static_batch_size is None:
            nw_val = tf.gather_nd(img, tf.cast(grid_nw, 'int32'))
        else:
            nw_val = self._pseudo_gather_nd(img, grid_nw)
        nw_val = nw_mask * nw_val

        grid_ne = tf.concat([b, y_n, x_e], 3)
        grid_ne = ne_mask * grid_ne
        if self.static_batch_size is None:
            ne_val = tf.gather_nd(img, tf.cast(grid_ne, 'int32'))
        else:
            ne_val = self._pseudo_gather_nd(img, grid_ne)
        ne_val = ne_mask * ne_val

        grid_sw = tf.concat([b, y_s, x_w], 3)
        grid_sw = sw_mask * grid_sw
        if self.static_batch_size is None:
            sw_val = tf.gather_nd(img, tf.cast(grid_sw, 'int32'))
        else:
            sw_val = self._pseudo_gather_nd(img, grid_sw)
        sw_val = sw_mask * sw_val

        grid_se = tf.concat([b, y_s, x_e], 3)
        grid_se = se_mask * grid_se
        if self.static_batch_size is None:
            se_val = tf.gather_nd(img, tf.cast(grid_se, 'int32'))
        else:
            se_val = self._pseudo_gather_nd(img, grid_se)
        se_val = se_mask * se_val

        out = nw * nw_val + sw * sw_val + ne * ne_val + se * se_val
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[1][1], input_shape[1][2], input_shape[0][3]

    def build(self, input_shape):
        img_shape = input_shape[0]
        grid_shape = input_shape[1]
        coefs = []
        for i in range(len(img_shape) - 1):
            coefs.append(int(tf.reduce_prod(img_shape[i+1:-1]).numpy()))
        self.i_H, self.i_W = grid_shape[1], grid_shape[2]
        if self.static_batch_size is not None:
            self.brange = tf.cast(tf.tile(tf.reshape(tf.range(self.static_batch_size), (-1,1,1,1)), (1, self.i_H, self.i_W, 1)), K.floatx()).numpy()
            self.coefs = coefs
            self.img_shape = (*img_shape,)
            self.final_shape = (self.static_batch_size, *grid_shape[1:-1], img_shape[-1])
        self.H, self.W = tf.cast(grid_shape[1], K.floatx()), tf.cast(grid_shape[2], K.floatx())
        self.iH, self.iW = tf.cast(img_shape[1], "int32"), tf.cast(img_shape[2], "int32")
        super(GridSample, self).build(input_shape)
    
    def get_config(self):
        config = super(GridSample, self).get_config()
        config.update({"static_batch_size": self.static_batch_size})
        return config


class FormHeatmap(layers.Layer):
    def __init__(self, spatial_size, num_kp, **kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(FormHeatmap, self).__init__(**kwargs)

    def call(self, x):
        h, w = self.spatial_size
        zeros = tf.reduce_sum(0 * x, 1)
        zeros = tf.reshape(zeros, (-1, 1, h * w))
        heatmap = tf.concat([zeros, x], 1)  # B 11 H*W
        heatmap = tf.reshape(heatmap, (-1, h * w, 1))  # B 11 HW1
        return heatmap

    def build(self, input_shape):
        super(FormHeatmap, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, self.num_kp + 1, self.spatial_size[0] * self.spatial_size[1], 1)

    def get_config(self):
        config = super(FormHeatmap, self).get_config()
        config.update({"spatial_size": self.spatial_size, "num_kp": self.num_kp})
        return config


def DownBlock2d(x, out_features, kernel_size=(3, 3), name=""):
    out = layers.Conv2D(out_features, kernel_size, strides=(1, 1), padding="same", name=name + "conv")(x)
    out = layers.BatchNormalization(trainable=False, momentum=0.1, epsilon=1e-05, name=name + "norm")(out)
    out = layers.Activation("relu")(out)
    out = layers.AveragePooling2D(pool_size=(2, 2))(out)
    return out


def UpBlock2d(x, out_features, kernel_size=(3, 3), name="", static_batch_size=None):
    out = Interpolate((2, 2), static_batch_size=static_batch_size)(x)
    out = layers.Conv2D(out_features, kernel_size, strides=(1, 1), padding="same", name=name + "conv")(out)
    out = layers.BatchNormalization(trainable=False, momentum=0.1, epsilon=1e-05, name=name + "norm")(out)
    out = layers.Activation("relu")(out)
    return out


def SameBlock2d(x, out_features, kernel_size=(7, 7), name=""):
    out = layers.Conv2D(out_features, kernel_size, padding="same", strides=(1, 1), name=name + "conv")(x)
    out = layers.BatchNormalization(trainable=False, momentum=0.1, epsilon=1e-05, name=name + "norm")(out)
    out = layers.Activation("relu")(out)
    return out


def ResBlock2d(x, out_features, kernel_size=(3, 3), name=""):
    out = layers.BatchNormalization(trainable=False, momentum=0.1, epsilon=1e-05, name=name + "norm1")(x)
    out = layers.Activation("relu")(out)
    out = layers.Conv2D(out_features, kernel_size, padding="same", strides=(1, 1), name=name + "conv1")(out)
    out = layers.BatchNormalization(trainable=False, momentum=0.1, epsilon=1e-05, name=name + "norm2")(out)
    out = layers.Activation("relu")(out)
    out = layers.Conv2D(out_features, kernel_size, strides=(1, 1), padding="same", name=name + "conv2")(out)
    out = layers.Add()([out, x])
    return out


def create_heatmap_representation(kp_driving, kp_source, h, w, num_kp, static_batch_size=None):
    gaussian_driving = KpToGaussian((h, w), num_kp, static_batch_size=static_batch_size)(kp_driving)
    gaussian_source = KpToGaussian((h, w), num_kp, static_batch_size=static_batch_size)(kp_source)
    if static_batch_size is None:
        gaussian_source = layers.Lambda(lambda l: tf.tile(l[0], (tf.shape(l[1])[0], 1, 1)))([gaussian_source, gaussian_driving])
    else:
        gaussian_source = layers.Lambda(lambda l: tf.tile(l[0], (static_batch_size, 1, 1)))([gaussian_source, gaussian_driving])
    heatmap = layers.Subtract()([gaussian_driving, gaussian_source])  # B(KP) H * W
    heatmap = FormHeatmap((h, w), num_kp)(heatmap)
    return heatmap

def dense_motion(
    source,
    source_image_scaled,
    kp_driving,
    kp_driving_jacobian,
    kp_source,
    kp_source_jacobian,
    shape,
    num_channels=3,
    num_kp=10,
    estimate_occlusion_map=True,
    estimate_jacobian=True,
    block_expansion=64,
    max_features=1024,
    num_blocks=5,
    scale_factor=0.25,
    static_batch_size=None,
):
    if source_image_scaled is not None:
        source = source_image_scaled
    elif scale_factor != 1:
        source = AntiAliasInterpolation2d(num_channels, scale_factor, static_batch_size=1)(source)

    h, w = shape
    h = int(h * scale_factor)
    w = int(w * scale_factor)
    heatmap_representation = create_heatmap_representation(kp_driving, kp_source, h, w, num_kp, static_batch_size=static_batch_size)  # B*11 H * W 1
    if not estimate_jacobian:
        kp_driving_jacobian, kp_source_jacobian = None, None
    sparse_motion = SparseMotion((h, w), num_kp, estimate_jacobian, static_batch_size=static_batch_size)([kp_driving, kp_driving_jacobian, kp_source, kp_source_jacobian])  # B 11 H*W 2
    deformed_source = Deform((h, w), num_channels, num_kp, static_batch_size=static_batch_size)([source, sparse_motion])  # B*11 H*W C
    
    inp = layers.Concatenate(2)([heatmap_representation, deformed_source])  # B*11 H*W C+1
    inp = layers.Lambda(lambda l: tf.reshape(l, (-1, num_kp+1, h * w, num_channels+1)))(inp)
    inp = layers.Lambda(lambda l: tf.transpose(l, (0, 2, 1, 3)))(inp) #B HW 11 C+1
    inp = layers.Lambda(lambda l: tf.reshape(l, (-1, h, w, (num_kp + 1) * (num_channels + 1))), name='dense_motion_networkinputreshape')(inp)  # B H W 44
    x = inp
    l = []
    l.append(x)
    for i in range(num_blocks):
        l.append(DownBlock2d(l[-1], min(max_features, block_expansion * (2 ** (i + 1))), name="dense_motion_networkhourglassencoderdown_blocks" + str(i)))

    x = l.pop()
    for i in range(num_blocks)[::-1]:
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** (i))), name="dense_motion_networkhourglassdecoderup_blocks" + str(4 - i), static_batch_size=static_batch_size)
        skip = l.pop()
        x = layers.Concatenate(3)([x, skip])

    mask = layers.Conv2D(num_kp + 1, kernel_size=(7, 7), padding="same", name="dense_motion_networkmask")(x)  # BHW(11)
    mask = layers.Lambda(lambda l: keras.activations.softmax(l))(mask)
    outmask = mask
    
    mask_shape = mask.shape
    mask = layers.Lambda(lambda l: tf.reshape(l, (-1, mask_shape[1], mask_shape[2], num_kp + 1)), name='dense_motion_networkmaskreshape0')(mask)
    mh, mw = mask.shape[1], mask.shape[2]
    mask = layers.Lambda(lambda l: tf.reshape(l, (-1, mh * mw, num_kp + 1, 1)), name='dense_motion_networkmaskreshape1')(mask)
    mask = layers.Permute((2, 1, 3))(mask)

    sparse_motion = layers.Lambda(lambda l: tf.reshape(l, (-1, num_kp + 1, mh * mw, 2)), name='dense_motion_networkparsemotionreshape')(sparse_motion)
    deformation = layers.Multiply()([sparse_motion, mask])  # b 11 64 * 64 2
    deformation = layers.Lambda(lambda l: K.sum(l, axis=1))(deformation)  # b 64 * 64 2
    deformation = layers.Lambda(lambda l: tf.reshape(l, (-1, mh, mw, 2)), name='dense_motion_networkdeformationreshape')(deformation)
    
    if estimate_occlusion_map:
        occlusion_map = layers.Conv2D(1, kernel_size=(7, 7), padding="same", name="dense_motion_networkocclusion")(x)
        occlusion_map = layers.Activation("sigmoid")(occlusion_map)
    else:
        occlusion_map = None
    
    deformed_source = layers.Lambda(lambda l: tf.reshape(l, (-1, num_kp+1, h, w, num_channels)))(deformed_source)
    return deformation, occlusion_map, outmask, deformed_source


def build_kp_detector_base(
    checkpoint="./checkpoint/vox-cpk.pth.tar",
    frame_shape=(256, 256, 3),
    num_channels=3,
    num_kp=10,
    temperature=0.1,
    block_expansion=32,
    max_features=1024,
    scale_factor=0.25,
    num_blocks=5,
    estimate_jacobian=True,
    single_jacobian_map=False,
    pad=0,
    static_batch_size=None,
    prescale=False,
):
    inp = layers.Input(shape=(frame_shape[0], frame_shape[1], num_channels), dtype=K.floatx(), name="img")
    
    # downsample
    if scale_factor == 1:
        x = inp
    else:
        x = AntiAliasInterpolation2d(num_channels, scale_factor, static_batch_size=static_batch_size)(inp)
        source_image_scaled = x

    # encode
    l = []
    l.append(x)
    for i in range(num_blocks):
        l.append(DownBlock2d(l[-1], min(max_features, block_expansion * (2 ** (i + 1))), name="downblock" + str(i)))

    # decode
    x = l.pop()
    for i in range(num_blocks)[::-1]:
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** i)), name="upblock" + str(i), static_batch_size=static_batch_size)
        skip = l.pop()
        x = layers.Concatenate(3)([x, skip])
    feature_map = x
    
    # prediction
    out = layers.ZeroPadding2D(pad)(x)
    out = layers.Conv2D(num_kp, kernel_size=(7, 7))(x)
    
    # Heatmap, gaussian
    out, heatmap = GaussianToKpTail(temperature, (out.shape[1], out.shape[2]), num_kp, name="output")(out)
    out_dict = {'value':out}
    
    # estimate jacobian
    if estimate_jacobian:
        num_jacobian_map = 1 if single_jacobian_map else num_kp
        padded_feature_map = layers.ZeroPadding2D(pad)(feature_map)
        jacobian_map = layers.Conv2D(4 * num_jacobian_map, kernel_size=(7, 7), name="jacmap")(padded_feature_map)
        jm0_shape = jacobian_map.shape
        
        jacobian_map = layers.Lambda(lambda l: tf.reshape(l, (-1, jm0_shape[1], jm0_shape[2], num_jacobian_map, 4)), name='jacmapreshape0')(jacobian_map) #b h w num_kp 4, bhw14
        if single_jacobian_map:
            jacobian_map = layers.Lambda(lambda l: tf.tile(l, (1, 1, 1, num_kp, 1)))(jacobian_map)
        jm1_shape = jacobian_map.shape
        jacobian_map = layers.Lambda(lambda l: tf.reshape(l, (-1, jm1_shape[1] * jm1_shape[2] * num_jacobian_map, 4)), name='jacmapreshape1')(jacobian_map)
        heatmap = layers.Lambda(lambda l: tf.reshape(l, (-1, jm0_shape[1] * jm0_shape[2], num_kp, 1)))(heatmap)
        heatmap = layers.Lambda(lambda l: tf.tile(l, (1, 1, 1, 4)))(heatmap)
        heatmap_shape = heatmap.shape
        heatmap = layers.Lambda(lambda l: tf.reshape(l, (-1, heatmap_shape[1] * heatmap_shape[2], 4)), name='heatmapreshape')(heatmap)
        jacobian = layers.Multiply(name='jacobian_mul')([heatmap, jacobian_map])  # 1 h w 10 4
        jacobian = layers.Lambda(lambda l: tf.reshape(l, (-1, heatmap_shape[1], num_jacobian_map, 4)), name='jacobianreshape0')(jacobian)        
        jacobian = layers.Lambda(lambda l: tf.reduce_sum(l, 1))(jacobian)
        jacobian = layers.Lambda(lambda l: tf.reshape(l, (-1, num_jacobian_map, 2, 2)), name='jacobianreshape1')(jacobian)
        out_dict['jacobian'] = jacobian
        
    if prescale:
        out_dict['source_image_scaled'] = source_image_scaled 
    
    # make model
    model = keras.Model(inp, out_dict)
    model.trainable = False
    model.compile()
    
    if checkpoint is not None and '.pth.tar' in checkpoint:
        sd = load_torch_checkpoint(checkpoint)["kp_detector"]
        sd = list(sd.values())
        for layer in model.layers:
            if "Conv2D" in repr(layer):
                weight = sd.pop(0)
                weight = weight.transpose(2, 3, 1, 0)
                bias = sd.pop(0)
                layer.set_weights([weight, bias])
            if "BatchNormalization" in repr(layer):
                gamma = sd.pop(0)
                beta = sd.pop(0)
                mean = sd.pop(0)
                var = sd.pop(0)
                num_batches_tracked = sd.pop(0)
                layer.set_weights([gamma, beta, mean, var])
                
    return model


class KpDetector(tf.Module):
    def __init__(
        self,
        checkpoint="./checkpoint/vox-cpk.pth.tar",
        frame_shape=(256, 256, 3),
        num_channels=3,
        num_kp=10,
        temperature=0.1,
        block_expansion=32,
        max_features=1024,
        scale_factor=0.25,
        num_blocks=5,
        estimate_jacobian=True,
        single_jacobian_map=False,
        pad=0,
        static_batch_size=None,
        prescale=False,
        **kwargs,
    ):
        self.single_jacobian_map = single_jacobian_map
        self.kp_detector = build_kp_detector_base(
            checkpoint=checkpoint,
            frame_shape=frame_shape,
            num_channels=num_channels,
            num_kp=num_kp,
            temperature=temperature,
            block_expansion=block_expansion,
            max_features=max_features,
            scale_factor=scale_factor,
            num_blocks=num_blocks,
            estimate_jacobian=estimate_jacobian,
            single_jacobian_map=single_jacobian_map,
            pad=pad,
            static_batch_size=static_batch_size,
            prescale=prescale
        )
        self.__call__ = tf.function(input_signature=[tf.TensorSpec([static_batch_size, frame_shape[0], frame_shape[1], num_channels], K.floatx())])(self.__call__)
        super(KpDetector, self).__init__()
    
    def __call__(self, img):
        return self.kp_detector(img)


def build_kp_detector(checkpoint, static_batch_size=None, prescale=False, **kwargs):
    return KpDetector(checkpoint=checkpoint, static_batch_size=static_batch_size, prescale=prescale, **kwargs)


def build_generator_base(
    checkpoint="./checkpoint/vox-cpk.pth.tar",
    full_output=True,
    frame_shape=(256, 256, 3),
    num_channels=3,
    num_kp=10,
    estimate_jacobian=True,
    single_jacobian_map=False,
    block_expansion=64,
    max_features=512,
    num_down_blocks=2,
    num_bottleneck_blocks=6,
    estimate_occlusion_map=True,
    dense_motion_params={"block_expansion": 64, "max_features": 1024, "num_blocks": 5, "scale_factor": 0.25},
    static_batch_size=None,
    prescale=False,
):
    jacobian_number = 1 if single_jacobian_map else num_kp
    inp = layers.Input(shape=(frame_shape[0], frame_shape[1], num_channels), dtype=K.floatx(), name="source")
    driving_kp = layers.Input(shape=(num_kp, 2), dtype=K.floatx(), name="kp_driving")
    kp_driving_jacobian = layers.Input(shape=(jacobian_number, 2, 2), dtype=K.floatx(), name="kp_driving_jacobian")
    source_kp = layers.Input(shape=(num_kp, 2), dtype=K.floatx(), name="kp_source")
    kp_source_jacobian = layers.Input(shape=(jacobian_number, 2, 2), dtype=K.floatx(), name="kp_source_jacobian")
    if prescale and dense_motion_params is not None:
        scale_factor = dense_motion_params["scale_factor"]
        inp_scaled = layers.Input(shape=(int(frame_shape[0] * scale_factor), int(frame_shape[1] * scale_factor), num_channels), dtype=K.floatx(), name="source_image_scaled")
    else:
        inp_scaled = None

    # source_image only
    x = SameBlock2d(inp, block_expansion, name="first")
    for i in range(num_down_blocks):
        x = DownBlock2d(x, min(max_features, block_expansion * (2 ** (i + 1))), name="down_blocks" + str(i))
    
    mask, sparse_deformed, occlusion_map, deformed = None, None, None, None
    
    if dense_motion_params is not None:
        deformation, occlusion_map, mask, sparse_deformed = dense_motion(
            inp, inp_scaled, driving_kp, kp_driving_jacobian, source_kp, 
            kp_source_jacobian, (frame_shape[0], frame_shape[1]), 
            num_channels=num_channels, num_kp=num_kp,
            estimate_occlusion_map=estimate_occlusion_map,
            estimate_jacobian=estimate_jacobian, 
            static_batch_size=static_batch_size,
            **dense_motion_params
        )

        if deformation.shape[1] != x.shape[1] or deformation.shape[2] != x.shape[2]:
            deformation = BilinearInterpolate((x.shape[1], x.shape[2]), static_batch_size=static_batch_size)(deformation)
        if static_batch_size is None:
            x = layers.Lambda(lambda l: tf.tile(l[0], (tf.shape(l[1])[0], 1, 1, 1)), name="deformation_tile")([x, deformation])
        else:
            x = layers.Lambda(lambda l: tf.tile(l[0], (static_batch_size, 1, 1, 1)), name="deformation_tile")([x, deformation])
        x = GridSample(static_batch_size=static_batch_size)([x, deformation])
        
        if estimate_occlusion_map:
            if occlusion_map.shape[1] != x.shape[1] or occlusion_map.shape[2] != x.shape[2]:
                occlusion_map = BilinearInterpolate((x.shape[1], x.shape[2]), static_batch_size=static_batch_size)(occlusion_map)
            x = layers.Multiply(name="mult")([x, occlusion_map])
            
        if deformation.shape[1] != inp.shape[1] or deformation.shape[2] != inp.shape[2]:
            deformation = BilinearInterpolate((inp.shape[1], inp.shape[2]), static_batch_size=static_batch_size)(deformation)
        if static_batch_size is None:
            to_deform = layers.Lambda(lambda l: tf.tile(l[0], (tf.shape(l[1])[0], 1, 1, 1)), name="source_deformation_tile")([inp, deformation])
        else:
            to_deform = layers.Lambda(lambda l: tf.tile(l[0], (static_batch_size, 1, 1, 1)), name="source_deformation_tile")([inp, deformation])
        deformed = GridSample(static_batch_size=static_batch_size)([to_deform, deformation])
        
    for i in range(num_bottleneck_blocks):
        x = ResBlock2d(x, min(max_features, block_expansion * (2 ** num_down_blocks)), name="bottleneckr" + str(i))

    for i in range(num_down_blocks):
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1))), name="up_blocks" + str(i), static_batch_size=static_batch_size)

    out = layers.Conv2D(num_channels, kernel_size=(7, 7), padding="same", name="final", activation="sigmoid")(x)
    
    out_dict = {'prediction':out} 
    
    
    if full_output:
        for n, t in zip(['mask', 'sparse_deformed', 'occlusion_map', 'deformed'], [mask, sparse_deformed, occlusion_map, deformed]):
            if t is not None:
                out_dict[n] = t
    
    if estimate_jacobian:
        inputs = [inp, driving_kp, kp_driving_jacobian, source_kp, kp_source_jacobian]
    else:
        inputs = [inp, driving_kp, source_kp]
    if prescale and dense_motion_params is not None:
        inputs.append(inp_scaled)
    
    model = keras.Model(inputs, out_dict)
    model.trainable = False
    model.compile()

    sd = load_torch_checkpoint(checkpoint)["generator"]
    sd = list(sd.items())
    while len(sd) > 0:
        k, v = sd.pop(0)
        if ".norm" in k:
            layer = model.get_layer(name="".join(k.split(".")[:-1]).replace(".", ""))
            gamma = v
            _, beta = sd.pop(0)
            _, mean = sd.pop(0)
            _, var = sd.pop(0)
            num_batches_tracked, _ = sd.pop(0)
            layer.set_weights([gamma, beta, mean, var])
        if ".conv" in k or ".mask" in k or ".occlusion" in k or "final" in k:
            layer = model.get_layer(name="".join(k.split(".")[:-1]).replace(".", ""))
            weight = v
            _, bias = sd.pop(0)
            weight = weight.transpose(2, 3, 1, 0)
            layer.set_weights([weight, bias])
    
    return model


class Generator(tf.Module):
    def __init__(
        self,
        checkpoint="./checkpoint/vox-cpk.pth.tar",
        full_output=True,
        frame_shape=(256, 256, 3),
        num_channels=3,
        num_kp=10,
        estimate_jacobian=True,
        single_jacobian_map=False,
        block_expansion=64,
        max_features=512,
        num_down_blocks=2,
        num_bottleneck_blocks=6,
        estimate_occlusion_map=True,
        dense_motion_params={"block_expansion": 64, "max_features": 1024, "num_blocks": 5, "scale_factor": 0.25},
        static_batch_size=None,
        prescale=False,
        **kwargs,
    ):
        jacobian_number = 1 if single_jacobian_map else num_kp
        self.generator = build_generator_base(
            checkpoint=checkpoint,
            full_output=full_output,
            frame_shape=frame_shape,
            num_kp=num_kp,
            num_channels=num_channels,
            estimate_jacobian=estimate_jacobian,
            single_jacobian_map=single_jacobian_map,
            block_expansion=block_expansion,
            max_features=max_features,
            num_down_blocks=num_down_blocks,
            num_bottleneck_blocks=num_bottleneck_blocks,
            estimate_occlusion_map=estimate_occlusion_map,
            dense_motion_params=dense_motion_params,
            static_batch_size=static_batch_size,
            prescale=prescale,
        )
        input_signature=[
                tf.TensorSpec([1, frame_shape[0], frame_shape[1], num_channels], K.floatx()),
                tf.TensorSpec([static_batch_size, num_kp, 2], K.floatx()),
                tf.TensorSpec([static_batch_size, jacobian_number, 2, 2], K.floatx()),
                tf.TensorSpec([1, num_kp, 2], K.floatx()),
                tf.TensorSpec([1, jacobian_number, 2, 2], K.floatx()),
            ]
        if prescale and dense_motion_params is not None:
            scale_factor = dense_motion_params["scale_factor"]
            input_signature.append(tf.TensorSpec([1, int(frame_shape[0] * scale_factor), int(frame_shape[1] * scale_factor), num_channels], K.floatx(), name="source_image_scaled"))
        if estimate_jacobian:
            self.__call__ = tf.function(input_signature=input_signature)(self.__call__)
        else:
            for to_remove in [input_signature[2], input_signature[4]]:
                input_signature.remove(to_remove)
            self.__call__ = tf.function(input_signature=input_signature)(self.call_nojacobian)
        super(Generator, self).__init__()
        
    
    def call_nojacobian(self, source_image, kp_driving, kp_source, *args):
        return self.generator([source_image, kp_driving, kp_source, *args])

    def __call__(self, source_image, kp_driving, kp_driving_jacobian, kp_source, kp_source_jacobian, *args):
        return self.generator([source_image, kp_driving, kp_driving_jacobian, kp_source, kp_source_jacobian, *args])
            


def build_generator(checkpoint, full_output=True, static_batch_size=None, prescale=False, **kwargs):
    return Generator(checkpoint=checkpoint, full_output=full_output, static_batch_size=static_batch_size, prescale=prescale, **kwargs)


class ProcessKpDriving(tf.Module):
    def __init__(self, num_kp, estimate_jacobian=True, single_jacobian_map=False, static_batch_size=None, hardcode=None):
        self.num_kp = num_kp
        self.estimate_jacobian = estimate_jacobian
        jacobian_number = 1 if single_jacobian_map else self.num_kp
        self.jacobian_number = jacobian_number
        self.static_batch_size = static_batch_size
        self.n = tf.cast(num_kp, 'int32').numpy()
        self.L = tf.cast(1, 'int32').numpy() # I noticed very late that convex_hull_area's only ever called on batch-1 kp tensors. Change this to tf.shape(X)[0]/static_batch_size if you need to reuse with >1 batches.
        self.j = [tf.repeat(tf.cast(x, 'int32'), self.L).numpy() for x in range(num_kp)]
        self.rng = tf.zeros(1, dtype='int32').numpy() # And change this to tf.range(L)
        self.sqrng = self.rng # And this to tf.range(L * L)
        sqrng = self.sqrng
        left = tf.reshape(sqrng, (self.L, self.L))
        right = tf.transpose(left)
        self.eye = tf.cast((left==right), K.floatx()).numpy()
        self.L_ones = tf.ones((self.L, 1), dtype=K.floatx()).numpy()
        if static_batch_size is not None:
            self.brange = tf.range(static_batch_size)
            self.bsqrange = tf.range(static_batch_size * static_batch_size)
        input_signature=[
                tf.TensorSpec([static_batch_size, num_kp, 2], K.floatx()),
                tf.TensorSpec([static_batch_size, jacobian_number, 2, 2], K.floatx()),
                tf.TensorSpec([1, num_kp, 2], K.floatx()),
                tf.TensorSpec([1, jacobian_number, 2, 2], K.floatx()),
                tf.TensorSpec([1, num_kp, 2], K.floatx()),
                tf.TensorSpec([1, jacobian_number, 2, 2], K.floatx()),
                tf.TensorSpec((), K.floatx()),
                tf.TensorSpec((), K.floatx()),
            ]
        if estimate_jacobian:
            self.__call__ = tf.function(input_signature=input_signature)(self.__call__)
        else:
            for to_remove in [input_signature[1], input_signature[3], input_signature[5], input_signature[6]]:
                input_signature.remove(to_remove)
            self.__call__ = tf.function(input_signature=input_signature)(self.call_nojacobian)
        self.hardcode = hardcode
        super(ProcessKpDriving, self).__init__()

    def __call__(
        self, kp_driving, kp_driving_jacobian, 
        kp_driving_initial, kp_driving_initial_jacobian,
        kp_source, kp_source_jacobian, 
        use_relative_jacobian, adapt_movement_scale
    ):
        kp_new = self.calculate_new_driving(kp_driving, kp_driving_initial, kp_source, adapt_movement_scale)
        kp_new_jacobian = self.calculate_new_jacobian(kp_driving_jacobian, kp_driving_initial_jacobian, kp_source_jacobian,
                                use_relative_jacobian)
        return {'value':kp_new, 'jacobian':kp_new_jacobian}
    
    def call_nojacobian(self, kp_driving, kp_driving_jacobian, 
        kp_driving_initial, kp_source, adapt_movement_scale):
        kp_new = self.calculate_new_driving(kp_driving, kp_driving_initial, kp_source, adapt_movement_scale)
        return {'value':kp_new, 'jacobian':kp_new_jacobian}
       
    def calculate_new_driving(self, kp_driving, kp_driving_initial, kp_source, adapt_movement_scale):
        if self.hardcode is not None and self.hardcode[1]=='0':
            scale = 1.0
        else:
            source_area = self.convex_hull_area(kp_source)        
            driving_area = self.convex_hull_area(kp_driving_initial)
            scale = tf.sqrt(source_area) / tf.sqrt(driving_area)
            ones = scale * 0.0 + 1.0
            scale = (adapt_movement_scale * scale + (1.0 - adapt_movement_scale) * ones)[None][None][None]
        kp_value_diff = kp_driving - kp_driving_initial
        kp_value_diff = kp_value_diff * scale

        kp_new = kp_value_diff + kp_source
        ones = kp_new * 0.0 + 1.0
        
        kp_new = tf.reshape(kp_new, (-1, self.num_kp, 2))
        return kp_new
    
    def calculate_new_jacobian(self, kp_driving_jacobian, kp_driving_initial_jacobian, kp_source_jacobian, use_relative_jacobian):
        if self.hardcode is not None and self.hardcode[0] == '0':
            return kp_driving_jacobian
        inv_kp_driving_initial_jacobian = self.batch_batch_four_by_four_inv(kp_driving_initial_jacobian)
        inv_kp_driving_initial_jacobian = tf.reshape(inv_kp_driving_initial_jacobian, (-1, 2, 2))
        kp_source_jacobian = tf.reshape(kp_source_jacobian, (-1, 2, 2))
        if self.static_batch_size is None:
            jacobian_diff = kp_driving_jacobian @ inv_kp_driving_initial_jacobian
            kp_new_jacobian = jacobian_diff @ kp_source_jacobian            
        else:
            jacobian_diff = self.jacobian_pseudo_batch_matmul(kp_driving_jacobian, inv_kp_driving_initial_jacobian)
            kp_new_jacobian = self.jacobian_pseudo_batch_matmul(jacobian_diff, kp_source_jacobian)
        if self.hardcode is not None:
            return kp_new_jacobian
        kp_new_jacobian = kp_new_jacobian * use_relative_jacobian + kp_driving_jacobian * (1.0 - use_relative_jacobian) 
        return kp_new_jacobian
    
    def jacobian_pseudo_batch_matmul(self, A, v):
        res = []
        for i in range(self.jacobian_number):
            left = tf.reshape(A[:, i:i+1, :, :], (self.static_batch_size * 2, 2))
            right = tf.reshape(v[None][:, i:i+1], (2, 2))
            out = tf.nn.bias_add(left @ right, tf.ones(2, dtype=left.dtype) * 1e-30)
            res.append(tf.reshape(out, (1, self.static_batch_size * 2, 2))) # b 1 2 2
        return tf.reshape(tf.transpose(tf.reshape(tf.concat(res, 0), (self.num_kp, self.static_batch_size, 2, 2)), (1, 0, 2, 3)), (self.static_batch_size, self.jacobian_number, 2, 2))
    
    @tf.function
    def convex_hull_area(self, X):
        L = self.L
        L_ones = self.L_ones
        rng = self.rng
        sqrng = self.sqrng
        num_kp = self.num_kp
        n = self.n
        O = X * 0
        l = tf.argmin(X[:, :, 0], 1, output_type='int32')
        p = l
        for i in range(num_kp):
            rng_p_stack = tf.stack([rng, p])
            pt = tf.transpose(tf.stack([rng, p]), (1, 0))
            ind = tf.expand_dims(tf.gather_nd(X, pt), 1)
            O = tf.concat([O[:, 0:i], ind, O[:, i + 1 : n]], 1)
            q = (p + 1) % n
            qt = tf.transpose(tf.stack([rng, q]), (1, 0))
            for j in range(num_kp):
                b = (
                    (tf.gather_nd(X, qt)[:, 1] - tf.gather_nd(X, pt)[:, 1]) * (X[:, j, 0] - tf.gather_nd(X, qt)[:, 0])
                    - (tf.gather_nd(X, qt)[:, 0] - tf.gather_nd(X, pt)[:, 0]) * (X[:, j, 1] - tf.gather_nd(X, qt)[:, 1])
                ) < 0
                q = tf.where(b, self.j[j], q)
                qt = tf.transpose(tf.stack([rng, q]), (1, 0))
            p = tf.where((((q - l) < 1) & ((q - l) > -1)), p, q)
        u = tf.transpose(tf.concat([tf.expand_dims(O[:, :, 1][:, 1], 1), O[:, :, 1][:, 2:], tf.expand_dims(O[:, :, 1][:, 0], 1)], 1), (1, 0))
        k = tf.transpose(tf.concat([tf.expand_dims(O[:, :, 0][:, 1], 1), O[:, :, 0][:, 2:], tf.expand_dims(O[:, :, 0][:, 0], 1)], 1), (1, 0))        
        eye = self.eye
        inc = (eye * tf.tensordot(O[:, :, 0], u, 1)) @ L_ones - (eye * tf.tensordot(O[:, :, 1], k, 1)) @ L_ones
        area = 0.5 * tf.math.sqrt(inc * inc)[0]
        return area
    
    def batch_batch_four_by_four_inv(self, x):
        num_kp = self.num_kp
        a = x[:, :, 0:1, 0:1]
        b = x[:, :, 0:1, 1:2]
        c = x[:, :, 1:2, 0:1]
        d = x[:, :, 1:2, 1:2]
        dets = a * d - b * c
        dets = tf.reshape(dets, (-1, num_kp, 1, 1))
        d = tf.reshape(d, (-1, num_kp, 1))
        c = tf.reshape(c, (-1, num_kp, 1))
        b = tf.reshape(b, (-1, num_kp, 1))
        a = tf.reshape(a, (-1, num_kp, 1))
        row1 = tf.concat([d, (-1 * c)], 2)
        row2 = tf.concat([(-1 * b), a], 2)
        row1 = tf.reshape(row1, (-1, num_kp, 2, 1))
        row2 = tf.reshape(row2, (-1, num_kp, 2, 1))
        m = tf.concat([row1, row2], 3)        
        return m / dets



def build_process_kp_driving(num_kp=10, estimate_jacobian=True, single_jacobian_map=False, static_batch_size=None, hardcode=None, **kwargs):
    return ProcessKpDriving(num_kp, estimate_jacobian, single_jacobian_map, static_batch_size, hardcode)
