import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers

class AntiAliasInterpolation2d(layers.Layer):

    def __init__(self, channels, scale, **kwargs):
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1        
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ka = ka
        self.kb = kb
        self.groups = channels
        self.scale = scale
        
        super(AntiAliasInterpolation2d,self).__init__(**kwargs)
    
    def build(self, input_shape):
        kernel_size = self.kernel_size
        sigma = self.sigma
        kernel = tf.ones(1)
        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        meshgrids = tf.meshgrid(*(tf.cast(tf.range(n),'float') for n in kernel_size))
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel = kernel * tf.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))     
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.reshape(kernel,(1, 1, *kernel.shape))
        kernel = tf.tile(kernel,(self.groups,1,1,1))
        kernel = tf.transpose(kernel,(2,3,1,0))
        self.kernel = self.add_weight('kernel',kernel.shape,trainable=False)
        self.set_weights([kernel])
        super(AntiAliasInterpolation2d, self).build(input_shape)
    
    @tf.autograph.experimental.do_not_convert
    def channel_slice(self, x, i):
        return x[:,:,:,i]
    
    def call(self, x):
        out = layers.ZeroPadding2D((self.ka,self.kb))(x)
        outputs = [0] * self.groups
        for i in range(self.groups):
            k = tf.expand_dims(self.channel_slice(self.kernel, i), 3)
            im = tf.expand_dims(self.channel_slice(out, i), 3)
            outputs[i] = tf.nn.conv2d(im,k,strides=(1,1,1,1),padding='VALID')
        out = tf.concat(outputs, 3)
        out=Interpolate((self.scale,self.scale))(out)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*self.scale, input_shape[2]*self.scale, input_shape[3])
        
    def get_config(self):
        config = super(AntiAliasInterpolation2d, self).get_config()
        config.update({'kernel_size':self.kernel_size,'groups':self.groups,'ka':self.ka,'kb':self.kb,'scale':self.scale})
        return config

def make_coordinate_grid(spatial_size, dtype):
    h,w = spatial_size
    h = tf.cast(h,dtype)
    w = tf.cast(w, dtype)
    x = tf.cast(tf.range(w),dtype=dtype)
    y = tf.cast(tf.range(h),dtype=dtype)
    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    yy = tf.tile(tf.reshape(y,(-1, 1)),(1, w))
    xx = tf.tile(tf.reshape(y,(1, -1)),(h, 1))
    meshed = tf.concat([tf.expand_dims(xx,2), tf.expand_dims(yy,2)], 2)
    return meshed
    
class GaussianToKpTail(layers.Layer):
    def __init__(self, temperature=0.1, spatial_size=(58,58), num_kp=10, **kwargs):
        self.temperature = temperature
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(GaussianToKpTail, self).__init__(**kwargs)
    
    def call(self, x):
        x = tf.cast(x, 'float32')
        out = tf.reshape(x,(-1,self.spatial_size[0]*self.spatial_size[1],self.num_kp)) #B (H*W) 3
        out = keras.activations.softmax(out/self.temperature, axis=1) #0.1 is temperature
        heatmap = tf.reshape(out,(-1,self.spatial_size[0],self.spatial_size[1],self.num_kp))
        heatmap2 = tf.transpose(heatmap,(0,3,1,2))
        heatmap2 = tf.expand_dims(heatmap2,4)
        heatmap2 = tf.tile(heatmap2, (1,1,1,1,2))
        grid = self.grid
        grid = tf.tile(grid, (1,10,1,1,1))
        heatmap2 = tf.reshape(heatmap2, (-1,self.num_kp * self.spatial_size[0] * self.spatial_size[1], 2))
        grid = tf.reshape(grid, (-1,self.num_kp * self.spatial_size[0] * self.spatial_size[1], 2))
        mult = heatmap2 * grid
        mult = tf.reshape(mult, (-1, self.num_kp, self.spatial_size[0], self.spatial_size[1], 2))
        value = tf.reduce_sum(mult,(2,3))
        #return value, heatmap
        return value
        
    def build(self, input_shape):
        self.grid = make_coordinate_grid(self.spatial_size, 'float32')[None][None]
        super(GaussianToKpTail, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[3],2)
        
    def get_config(self):
        config = super(GaussianToKpTail,self).get_config()
        config.update({'temperature':self.temperature})
        return config

class SparseMotion(layers.Layer): 
    def __init__(self,spatial_size,num_kp,**kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(SparseMotion,self).__init__(**kwargs)
    
    def call(self, x):
        bs = tf.shape(x[0])[0]
        h, w = self.spatial_size
        identity_grid = self.grid #hw2
        #(lambda l: [l,tf.io.write_file('../../spmx.npy',tf.io.serialize_tensor(l))][0])(x)
        coordinate_grid = identity_grid - tf.expand_dims(tf.expand_dims(x[0],2),2) #b 10 1 1 2
        c = x[1] * -1.0
        mult = tf.tile(tf.reshape(c, (-1,self.num_kp,1,1,2)),(bs,1,h,w,1))
        driving_to_source = coordinate_grid - mult  #if multiple works, coordinate_grid will be b 10 1 1 2 and the right part will be 1 10 1 1 2
        identity_grid = tf.tile(identity_grid,(bs,1,1,1,1))
        sparse_motions = tf.concat([identity_grid,driving_to_source],1)
        return sparse_motions
    
    def build(self, input_shape):
        self.grid = make_coordinate_grid(self.spatial_size, 'float32')[None][None]
        super(SparseMotion, self).build(input_shape)
    
    def compute_output_shape(self,input_shape):
        return (None,self.num_kp+1,self.spatial_size[0],self.spatial_size[1],2)
        
    def get_config(self):
        config = super(SparseMotion,self).get_config()
        config.update({'spatial_size':self.spatial_size, 'num_kp':self.num_kp})
        return config


class Deform(layers.Layer):
    def __init__(self,spatial_size,channels,num_kp,**kwargs):
        self.spatial_size = spatial_size
        self.channels = channels
        self.num_kp = num_kp
        super(Deform,self).__init__(**kwargs)
    
    def call(self, x):
        bs = tf.shape(x[1])[0]
        h, w = self.spatial_size
        source_repeat = tf.tile(tf.expand_dims(tf.expand_dims(x[0],1),1),(bs,self.num_kp+1,1,1,1,1))       
        #source_repeat = tf.reshape(source_repeat,(bs * (self.num_kp+1), h, w, self.channels))
        source_repeat = tf.reshape(source_repeat,(-1, h, w, self.channels))
        #sparse_motions = tf.reshape(x[1], (bs * (self.num_kp+1), h, w, 2))
        sparse_motions = tf.reshape(x[1], (-1, h, w, 2))
        sparse_deformed = GridSample()([source_repeat,sparse_motions])
        sparse_deformed = tf.reshape(sparse_deformed,(-1, self.num_kp+1, h, w,self.channels))
        return sparse_deformed
    
    def build(self, input_shape):
        super(Deform, self).build(input_shape)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0][0],input_shape[1][1],self.spatial_size[0],self.spatial_size[1],self.channels)
        
    def get_config(self):
        config = super(Deform,self).get_config()
        config.update({'spatial_size':self.spatial_size,'channels':self.channels,'num_kp':self.num_kp})
        return config


class KpToGaussian(layers.Layer):
    def __init__(self,spatial_size,num_kp,**kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(KpToGaussian,self).__init__(**kwargs)
    
    def call(self, x):
        mean = tf.cast(x, 'float32')
        kp_variance = tf.cast(0.01, 'float32')
        grid = self.grid #HW2
        shape = (1,1,self.spatial_size[0],self.spatial_size[1],2) 
        grid = tf.tile(grid, (1,10,1,1,1)) #B,10,256,256,2
        mean = tf.expand_dims(tf.expand_dims(mean,2),2) #B 10 1 1 2
        mean_sub = grid - mean
        mean_sub = layers.Reshape((self.num_kp*self.spatial_size[0]*self.spatial_size[1], 2))(mean_sub)
        #mean_sub = tf.reshape(mean_sub, (-1, self.num_kp*self.spatial_size[0]*self.spatial_size[1], 2))
        #mean_sub = tf.ensure_shape(mean_sub, (None, self.num_kp*self.spatial_size[0]*self.spatial_size[1], 2))
        mean_sub =  tf.multiply(mean_sub, mean_sub)
        mean_sub = tf.reshape(mean_sub, (-1, self.num_kp, self.spatial_size[0], self.spatial_size[1], 2))
        out = tf.math.exp(-0.5 * tf.reduce_sum(mean_sub, -1) / kp_variance)
        return out
    
    def build(self, input_shape):
        self.grid = make_coordinate_grid(self.spatial_size, 'float32')[None][None]
        super(KpToGaussian, self).build(input_shape)
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.num_kp,self.spatial_size[0],self.spatial_size[1])
        
    def get_config(self):
        config = super(KpToGaussian,self).get_config()
        config.update({'spatial_size':self.spatial_size,'num_kp':self.num_kp})
        return config
    
class Interpolate(layers.Layer):
    def __init__(self,scale_factor,**kwargs):
        self.scale_factor=tuple(float(x) for x in scale_factor)
        super(Interpolate,self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(Interpolate, self).build(input_shape)
    
    def call(self,img):
        scale_factor = self.scale_factor
        y_max = tf.floor(img.shape[1]*scale_factor[0])
        x_max = tf.floor(img.shape[2]*scale_factor[1])
        x_scale = img.shape[2]/x_max
        y_scale = img.shape[1]/y_max
        N,H,W = tf.shape(img)[0],img.shape[1],img.shape[2]
        H_f,W_f = tf.cast(H,'float32'),tf.cast(W,'float32')
        
        grid = tf.cast(tf.transpose(tf.stack(tf.meshgrid(tf.range(x_max),tf.range(y_max))[::-1]),(1,2,0)),'float32')
        grid_shape = (1,*grid.shape)
        flat_grid = tf.reshape(grid,(-1,2))
        flat_grid = tf.stack(([(tf.math.minimum(tf.floor((flat_grid[...,0])* y_scale), H_f - 1)),tf.math.minimum(tf.floor((flat_grid[...,1]) * x_scale), W_f - 1)]))
        flat_grid = tf.transpose(flat_grid,(1,0))
        grid = tf.reshape(flat_grid,grid_shape)
        grid = tf.cast(grid,'int32')
        
        batch_range = tf.range(N)
        g = tf.cast(tf.expand_dims(tf.tile(tf.reshape(batch_range,(-1,1,1)),(1,y_max,x_max)),3),'int32')
        grid = tf.tile(tf.reshape(grid,(1,y_max,x_max,2)),(N,1,1,1))
        grid = tf.concat([g,grid],3)
        
        out = tf.gather_nd(img,grid)   
        return out
        
    def compute_output_shape(self,input_shape):
        scale_factor = self.scale_factor
        return input_shape[0],input_shape[1]*scale_factor[0],input_shape[2]*scale_factor[1],input_shape[3]
        
    def get_config(self):
        config = super(Interpolate, self).get_config()
        config.update({'scale_factor':self.scale_factor})
        return config
        
class BilinearInterpolate(layers.Layer):
    def __init__(self,size,**kwargs):
        self.size=size
        super(BilinearInterpolate,self).__init__(**kwargs)
    
    def build(self, input_shape):
        super(BilinearInterpolate, self).build(input_shape)
    
    def call(self,img):
        size = self.size
        x_scale = tf.cast(img.shape[2]/size[1],'float32')
        y_scale = tf.cast(img.shape[1]/size[0],'float32')
        N,H,W = tf.shape(img)[0],img.shape[1],img.shape[2]
        H_f,W_f = tf.cast(H,'float32'),tf.cast(W,'float32')
        zeros = tf.zeros((size[0],size[1]),dtype='int32')
        ones = tf.expand_dims(tf.ones((size[0],size[1]),dtype='int32'),2)
        
        grid = tf.cast(tf.transpose(tf.stack(tf.meshgrid(tf.range(size[1]),tf.range(size[0]))[::-1]),(1,2,0)),'float32')
        grid_shape = (size[1],size[0],2)
        flat_grid = tf.reshape(grid,(-1,2))
        flat_grid = tf.stack([(flat_grid[:,0]+0.5) * y_scale - 0.5, (flat_grid[:,1]+0.5) * x_scale -0.5])
        flat_grid = tf.transpose(flat_grid,(1,0))
        flat_grid = tf.math.maximum(flat_grid,0)
        grid = tf.reshape(flat_grid,grid_shape)
        grid_int = tf.cast(tf.floor(grid),'int32')
        
        y = tf.cast(tf.transpose(tf.stack([tf.cast(grid_int[...,0] < W - 1,'int32'),zeros]),(1,2,0)), 'int32')
        x = tf.cast(tf.transpose(tf.stack([zeros,tf.cast(grid_int[...,1] < H - 1,'int32')]),(1,2,0)), 'int32')
        batch_range = tf.range(N)
        g = tf.cast(tf.expand_dims(tf.tile(tf.reshape(batch_range,(-1,1,1)),(1,size[0],size[1])),3),'int32')
        
        grid00 = grid_int
        grid01 = grid00 + x
        grid10 = grid00 + y
        grid11 = grid00 + y + x
        
        grid00 = tf.concat([g,tf.tile(tf.reshape(grid00,(1,size[0],size[1],2)),(N,1,1,1))],3)
        grid01 = tf.concat([g,tf.tile(tf.reshape(grid01,(1,size[0],size[1],2)),(N,1,1,1))],3)
        grid10 = tf.concat([g,tf.tile(tf.reshape(grid10,(1,size[0],size[1],2)),(N,1,1,1))],3)
        grid11 = tf.concat([g,tf.tile(tf.reshape(grid11,(1,size[0],size[1],2)),(N,1,1,1))],3)
        
        val00 = tf.gather_nd(img,grid00)
        val01 = tf.gather_nd(img,grid01)
        val10 = tf.gather_nd(img,grid10)
        val11 = tf.gather_nd(img,grid11)
        
        l = tf.tile(tf.reshape(grid,(1,size[0],size[1],2)),(N,1,1,1)) - tf.cast(grid_int,'float32')
        
        w0 = tf.expand_dims(l[...,1],3) 
        w1 = 1 - w0
        h0 = tf.expand_dims(l[...,0],3)
        h1 = 1 - h0
        
        return h1*(w1*val00 + w0*val01) + h0*(w1*val10 + w0*val11)
    
        
    def compute_output_shape(self,input_shape):
        return input_shape[0],self.size[0],self.size[1],input_shape[3]
        
    def get_config(self):
        config = super(BilinearInterpolate, self).get_config()
        config.update({'size':self.size})
        return config

class GridSample(layers.Layer):
    def __init__(self,**kwargs):
        super(GridSample,self).__init__(**kwargs)
    
    def call(self, data):
        img = data[0]
        grid = data[1]
        
        N,H,W = tf.cast(tf.shape(grid)[0],'float32'),tf.cast(grid.shape[1],'float32'),tf.cast(grid.shape[2],'float32')
        iH, iW = img.shape[1], img.shape[2]
        
        #extract x,y from grid
        x = tf.expand_dims(grid[...,0],3)
        y = tf.expand_dims(grid[...,1],3)
        
        #ComputeLocationBase
        x = (x+1)*(W/2) - 0.5
        y = (y+1)*(H/2) - 0.5
        
        #compute_interp_params
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
        
        x_w = tf.cast(x_w,'int32')
        y_n = tf.cast(y_n,'int32')
        x_e = tf.cast(x_e,'int32')
        y_s = tf.cast(y_s,'int32')
        
        #forward
        w_mask = (x_w>-1) & (x_w<iW) 
        n_mask = (y_n>-1) & (y_n<iH)
        e_mask = (x_e>-1) & (x_e<iW) 
        s_mask = (y_s>-1) & (y_s<iH)
        nw_mask = w_mask & n_mask
        ne_mask = e_mask & n_mask
        sw_mask = s_mask & w_mask
        se_mask = e_mask & s_mask
        
        #loop
        b = tf.range(N)
        b = tf.reshape(b,(-1,1,1,1))
        b = tf.tile(b, (1,H,W,1))
        b = tf.cast(b,'int32')
    
        grid_nw = tf.concat([b,y_n,x_w],3)
        grid_nw = tf.where(nw_mask,grid_nw,0)
        nw_val = tf.gather_nd(img,grid_nw)
        nw_val = tf.where(nw_mask,nw_val,0.0)
        
        grid_ne = tf.concat([b,y_n,x_e],3)
        grid_ne = tf.where(ne_mask,grid_ne,0)
        ne_val  = tf.gather_nd(img,grid_ne)
        ne_val  = tf.where(ne_mask,ne_val,0.0)
        
        grid_sw = tf.concat([b,y_s,x_w],3)
        grid_sw = tf.where(sw_mask,grid_sw,0)
        sw_val = tf.gather_nd(img,grid_sw)
        sw_val = tf.where(sw_mask,sw_val,0.0)
    
        grid_se = tf.concat([b,y_s,x_e],3)
        grid_se = tf.where(se_mask,grid_se,0)
        se_val = tf.gather_nd(img,grid_se)
        se_val = tf.where(se_mask,se_val,0.0)
        
        out = nw*nw_val + sw*sw_val + ne*ne_val + se*se_val
        return out

    def compute_output_shape(self,input_shape):
        return input_shape[0][0],input_shape[1][1],input_shape[1][2],input_shape[0][3] 
    
    def build(self,input_shape):
        super(GridSample, self).build(input_shape)
        
class FormHeatmap(layers.Layer):
    def __init__(self,spatial_size,num_kp,**kwargs):
        self.spatial_size = spatial_size
        self.num_kp = num_kp
        super(FormHeatmap,self).__init__(**kwargs)
    def call(self, x):
        h,w = self.spatial_size
        heatmap = tf.concat([tf.expand_dims(tf.reduce_sum(0*x,1),1), x],1) #B 11 HW
        heatmap = K.expand_dims(heatmap,4) # B 11 HW1
        return heatmap
    def build(self, input_shape):
        super(FormHeatmap, self).build(input_shape)
    
    
    def compute_output_shape(self,input_shape):
        1, self.num_kp+1,self.spatial_size[0], self.spatial_size[1], 1
    
    def get_config(self):
        config = super(FormHeatmap, self).get_config()
        config.update({'spatial_size':self.spatial_size})
        return config

    
def DownBlock2d(x, out_features, kernel_size=(3,3),name=''):
    out = layers.Conv2D(out_features,kernel_size,strides=(1,1),padding='same',name=name+'conv')(x)
    out = layers.BatchNormalization(trainable=False,momentum=0.1,epsilon=1e-05,name=name+'norm')(out)
    out = layers.Activation('relu')(out)
    out = layers.AveragePooling2D(pool_size=(2,2))(out)
    return out

def UpBlock2d(x, out_features, kernel_size=(3,3), name=''):
    out = Interpolate((2,2))(x)
    out = layers.Conv2D(out_features,kernel_size,strides=(1,1),padding='same',name=name+'conv')(out)
    out = layers.BatchNormalization(trainable=False,momentum=0.1,epsilon=1e-05,name=name+'norm')(out)
    out = layers.Activation('relu')(out)
    return out
    
def SameBlock2d(x, out_features, kernel_size=(7,7), name=''):
    out = layers.Conv2D(out_features,kernel_size,padding='same',strides=(1,1),name=name+'conv')(x)
    out = layers.BatchNormalization(trainable=False,momentum=0.1,epsilon=1e-05,name=name+'norm')(out)
    out = layers.Activation('relu')(out)
    return out
    
def ResBlock2d(x, out_features, kernel_size=(3,3), name=''):
    out = layers.BatchNormalization(trainable=False,momentum=0.1,epsilon=1e-05,name=name+'norm1')(x)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(out_features,kernel_size,padding='same',strides=(1,1),name=name+'conv1')(out)
    out = layers.BatchNormalization(trainable=False,momentum=0.1,epsilon=1e-05,name=name+'norm2')(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(out_features,kernel_size,strides=(1,1),padding='same',name=name+'conv2')(out)
    out = layers.Add()([out,x])
    return out

def create_heatmap_representation(kp_driving, kp_source,h,w,num_kp):
    gaussian_driving = KpToGaussian((h,w),num_kp)(kp_driving)
    gaussian_source = KpToGaussian((h,w),num_kp)(kp_source)
    gaussian_source = layers.Lambda(lambda l: tf.tile(l[0],(tf.shape(l[1])[0],1,1,1)))([gaussian_source,gaussian_driving])
    heatmap = layers.Subtract()([gaussian_driving,gaussian_source]) #B(KP)HW
    heatmap = FormHeatmap((h,w),num_kp)(heatmap)
    return heatmap

def reshape_series(x,h,w,c,kp):
    x = tf.expand_dims(x,0)
    x = tf.reshape(x, (-1,(kp+1),h,w,c+1))
    x = tf.transpose(x, (0,1,4,2,3))
    x = tf.reshape(x, (-1, (c+1)*(kp+1),h,w))
    x = tf.transpose(x,(0,2,3,1))
    return x
    
def dense_motion(source,kp_driving,kp_source,shape,num_channels=3,num_kp=10,estimate_jacobian=True,block_expansion=64,max_features=1024,num_blocks=5,scale_factor=0.25):
    if scale_factor != 1:
        source = AntiAliasInterpolation2d(num_channels,scale_factor)(source)
        
    h,w = shape
    h = int(h*scale_factor)
    w = int(w*scale_factor)
    heatmap_representation = create_heatmap_representation(kp_driving,kp_source,h,w,num_kp) #B 11 H W 1
    sparse_motion = SparseMotion((h,w),num_kp)([kp_driving,kp_source]) #B 11  H W 2 
    deformed_source = Deform((h,w),num_channels,num_kp)([source,sparse_motion]) #B 11 H W C
    
    inp = layers.Concatenate(4)([heatmap_representation,deformed_source]) #B 11 H W C+1
    inp = layers.Lambda(lambda l: reshape_series(l,h,w,num_channels,num_kp))(inp)
    inp = layers.Reshape((h,w,(num_kp+1)*(num_channels+1)))(inp) #won't reshape, but will assert the shape
    x = inp
    l = [] 
    l.append(x)
    for i in range(num_blocks):
        l.append(DownBlock2d(l[-1],min(max_features, block_expansion * (2 ** (i+1))),name='dense_motion_networkhourglassencoderdown_blocks'+str(i)))
        
    x = l.pop()
    for i in range(num_blocks)[::-1]:
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** (i))),name='dense_motion_networkhourglassdecoderup_blocks'+str(4-i))
        skip = l.pop()        
        x = layers.Concatenate(3)([x, skip])
    
    
   
    mask = layers.Conv2D(num_kp+1,kernel_size=(7, 7),padding='same',name='dense_motion_networkmask')(x) #BHW(11)
    mask = layers.Lambda(lambda l: keras.activations.softmax(l))(mask)    
    mask = layers.Permute((3,1,2))(mask)                                                   #B(11)HW
    
    
    mask = layers.Lambda(lambda l: K.expand_dims(l,2))(mask)

    
    
    sparse_motion = layers.Permute((1, 4, 2, 3))(sparse_motion)
    spatial = sparse_motion.shape[3], sparse_motion.shape[4]
    
    sparse_motion = layers.Reshape((num_kp+1, 2, sparse_motion.shape[3]*sparse_motion.shape[4]))(sparse_motion)
    mask = layers.Reshape((num_kp+1, 1, mask.shape[3]*mask.shape[4]))(mask)
    
    deformation = layers.Multiply()([sparse_motion, mask])
    deformation = layers.Reshape((num_kp+1, 2, spatial[0], spatial[1]))(deformation)
    deformation = layers.Lambda(lambda l: K.sum(l,axis=1))(deformation)
    deformation = layers.Permute((2,3,1))(deformation)
    
    occlusion_map = layers.Conv2D(1,kernel_size=(7, 7),padding='same', name='dense_motion_networkocclusion')(x)
    occlusion_map = layers.Activation('sigmoid')(occlusion_map) 
    
    return deformation, occlusion_map
    
def build_kp_detector_base(checkpoint='./checkpoint/vox-cpk.pth.tar',num_channels=3, num_kp=10, temperature=0.1, block_expansion=32, max_features=1024, scale_factor=0.25, num_blocks=5, estimate_jacobian=True, single_jacobian_map=False):
    inp = layers.Input(shape=(256,256,3),dtype='float32',name='img')
    #downsample
    x = AntiAliasInterpolation2d(num_channels,scale_factor)(inp) 
    
    #encode
    l = [] 
    l.append(x)
    for i in range(num_blocks):
        l.append(DownBlock2d(l[-1], min(max_features, block_expansion * (2 ** (i + 1))),name='downblock'+str(i)))   
    
    
    #decode
    x = l.pop()
    for i in range(num_blocks)[::-1]:
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** i)),name='upblock'+str(i))
        skip = l.pop()
        x = layers.Concatenate(3)([x, skip])
    feature_map  = x
    #prediction
    out = layers.Conv2D(num_kp,kernel_size=(7, 7))(x)
    #Heatmap, gaussian
    #out,heatmap = GaussianToKpTail(temperature,name='output')(out)
    out = GaussianToKpTail(temperature,(out.shape[1],out.shape[2]),num_kp,name='output')(out)
    #Jacobian
    #num_jacobian_map = 1 if single_jacobian_map else num_kp
    #jacobian_map = Conv2D(4*num_jacobian_map, kernel_size=(7,7),name='jacmap')(feature_map)
    #jacobian_map = Reshape((jacobian_map.shape[1], jacobian_map.shape[2],num_jacobian_map,4))(jacobian_map)
    #heatmap = Lambda(lambda l: tf.expand_dims(l,4))(heatmap)
    #jacobian = Multiply()([heatmap,jacobian_map])    #1 h w 10 4
    #jacobian = Lambda(lambda l: [l,tf.io.write_file('tf.npy',tf.io.serialize_tensor(l))][0])(jacobian)
    #jacobian = Permute((3,4,1,2))(jacobian) 
    #jacobian = Reshape((jacobian.shape[1], 4, -1))(jacobian)
    #jacobian = Lambda(lambda l: tf.reduce_sum(l,-1))(jacobian)
    #jacobian = Reshape((num_jacobian_map,2,2))(jacobian)
    #
    #model = Model(inp,[out,jacobian])
    model = keras.Model(inp,out)
    model.trainable = False
    model.compile('sgd','mse')
    import torch
    sd = torch.load(checkpoint,map_location=torch.device('cpu'))['kp_detector']
    sd = list(sd.values())       
    for layer in model.layers:
        if 'Conv2D' in repr(layer):
            weight = sd.pop(0).numpy()
            weight=weight.transpose(2,3,1,0)
            bias = sd.pop(0).numpy()
            layer.set_weights([weight,bias])
        if 'BatchNormalization' in repr(layer):
            gamma = sd.pop(0).numpy()
            beta = sd.pop(0).numpy()
            mean = sd.pop(0).numpy()
            var = sd.pop(0).numpy()
            num_batches_tracked = sd.pop(0).numpy()
            layer.set_weights([gamma,beta,mean,var])
    return model


class KpDetector(tf.Module):
    def __init__(self,checkpoint='./checkpoint/vox-cpk.pth.tar',num_channels=3, num_kp=10, temperature=0.1, block_expansion=32, max_features=1024, scale_factor=0.25, num_blocks=5, estimate_jacobian=True, single_jacobian_map=False):
        self.kp_detector = build_kp_detector_base(checkpoint=checkpoint,num_channels=num_channels,num_kp=num_kp,temperature=temperature,block_expansion=block_expansion,max_features=max_features,scale_factor=scale_factor,
                    num_blocks=num_blocks,estimate_jacobian=estimate_jacobian,single_jacobian_map=single_jacobian_map)
        self.__call__ = tf.function(input_signature=[tf.TensorSpec([None,256,256,num_channels],tf.float32)])(self.__call__)
        super(KpDetector,self).__init__()

    def __call__(self,img):
        return self.kp_detector(img,training=False)

def build_kp_detector(checkpoint='./checkpoint/vox-cpk.pth.tar',num_channels=3, num_kp=10, temperature=0.1, block_expansion=32, max_features=1024, scale_factor=0.25, num_blocks=5, estimate_jacobian=True, single_jacobian_map=False):
    return KpDetector(checkpoint=checkpoint,num_channels=num_channels,num_kp=num_kp,temperature=temperature,block_expansion=block_expansion,max_features=max_features,scale_factor=scale_factor,
                    num_blocks=num_blocks,estimate_jacobian=estimate_jacobian,single_jacobian_map=single_jacobian_map)

def build_generator_base(checkpoint='./checkpoint/vox-cpk.pth.tar', num_kp=10, num_channels=3, estimate_jacobian=True,
                block_expansion=64, max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                estimate_occlusion_map=True,dense_motion_params={'block_expansion':64, 'max_features':1024, 'num_blocks':5, 'scale_factor' :0.25}):
    inp = layers.Input(shape=(256,256,3),dtype='float32',name='source')
    driving_kp = layers.Input(shape=(num_kp,2),dtype='float32',name='kp_driving')
    source_kp = layers.Input(shape=(num_kp,2),dtype='float32',name='kp_source')
    
    #x = layers.Lambda(lambda l: [l[0],print(l[0].shape),print(l[1].shape),print(l[2].shape),input('generator t')][0])([inp,driving_kp,source_kp])
    
    x = SameBlock2d(inp, block_expansion, name='first')
    for i in range(num_down_blocks):
        x = DownBlock2d(x, min(max_features, block_expansion * (2 ** (i+1))),name='down_blocks'+str(i))
        
    deformation, occlusion_map = dense_motion(inp,driving_kp,source_kp,(256,256),num_channels=num_channels,num_kp=num_kp,estimate_jacobian=estimate_jacobian,**dense_motion_params)
    
    if deformation.shape[1] != x.shape[1] or deformation.shape[2] != x.shape[2]:
        deformation = BilinearInterpolate((x.shape[1],x.shape[2]))(deformation)
    if occlusion_map.shape[1] != x.shape[1] or occlusion_map.shape[2] != x.shape[2]:
        occlusion_map = BilinearInterpolate((x.shape[1],x.shape[2]))(occlusion_map)
    x = layers.Lambda(lambda l: tf.tile(l[0],(tf.shape(l[1])[0],1,1,1)),name='deformation_tile')([x,deformation])
    x = GridSample()([x,deformation]) 
    x = layers.Multiply(name='mult')([x,occlusion_map])
    for i in range(num_bottleneck_blocks):
        x = ResBlock2d(x, min(max_features, block_expansion * (2 ** num_down_blocks)),name='bottleneckr'+str(i))
        
    for i in range(num_down_blocks):
        x = UpBlock2d(x, min(max_features, block_expansion * (2 ** (num_down_blocks - i-1))),name='up_blocks'+str(i))
    
    x = layers.Conv2D(num_channels, kernel_size=(7, 7),padding='same',name='final')(x)
    out = layers.Activation('sigmoid',name='output')(x)
    model = keras.Model([inp,driving_kp,source_kp],out)
    model.trainable = False
    model.compile('sgd','mse')
    
    import torch
    sd = torch.load(checkpoint,map_location=torch.device('cpu'))['generator'] 
    sd = list(sd.items())
    while len(sd)>0:
        k,v = sd.pop(0)
        if '.norm' in k:
            layer =model.get_layer(name = ''.join(k.split('.')[:-1]).replace('.',''))
            gamma = v
            _, beta = sd.pop(0)
            _, mean = sd.pop(0)
            _, var = sd.pop(0)
            num_batches_tracked,_ = sd.pop(0)
            layer.set_weights([gamma.numpy(),beta.numpy(),mean.numpy(),var.numpy()])
        if '.conv' in k or '.mask' in k or '.occlusion' in k or 'final' in k:
            layer = model.get_layer(name = ''.join(k.split('.')[:-1]).replace('.',''))
            weight = v
            _,bias = sd.pop(0)
            weight=weight.permute(2,3,1,0)
            layer.set_weights([weight.numpy(),bias.numpy()])
        
    return model

class Generator(tf.Module):
    def __init__(self, checkpoint='./checkpoint/vox-cpk.pth.tar', num_kp=10, num_channels=3, estimate_jacobian=True,
                block_expansion=64, max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                estimate_occlusion_map=True,dense_motion_params={'block_expansion':64, 'max_features':1024, 'num_blocks':5, 'scale_factor' :0.25}):
        self.generator = build_generator_base(checkpoint=checkpoint,num_kp=num_kp,num_channels=num_channels,estimate_jacobian=estimate_jacobian,
                block_expansion=block_expansion,max_features=max_features,num_down_blocks=num_down_blocks,num_bottleneck_blocks=num_bottleneck_blocks,estimate_occlusion_map=estimate_occlusion_map,dense_motion_params=dense_motion_params)
        self.__call__ = tf.function(input_signature=[tf.TensorSpec([1,256,256,num_channels],tf.float32),tf.TensorSpec([None,num_kp,2],tf.float32),tf.TensorSpec([1,num_kp,2],tf.float32)])(self.__call__)
        super(Generator,self).__init__()
    
    def __call__(self, source_image, kp_driving, kp_source):
        return self.generator([source_image, kp_driving, kp_source],training=False)

def build_generator(checkpoint='./checkpoint/vox-cpk.pth.tar', num_kp=10, num_channels=3, estimate_jacobian=True,
                block_expansion=64, max_features=512, num_down_blocks=2, num_bottleneck_blocks=6,
                estimate_occlusion_map=True,dense_motion_params={'block_expansion':64, 'max_features':1024, 'num_blocks':5, 'scale_factor' :0.25}):
    return Generator(checkpoint=checkpoint,num_kp=num_kp,num_channels=num_channels,estimate_jacobian=estimate_jacobian,
                block_expansion=block_expansion,max_features=max_features,num_down_blocks=num_down_blocks,num_bottleneck_blocks=num_bottleneck_blocks,estimate_occlusion_map=estimate_occlusion_map,dense_motion_params=dense_motion_params)

class ProcessKpDriving(tf.Module):
    def __init__(self, num_kp):
        self.num_kp = num_kp
        self.__call__ = tf.function(input_signature=[tf.TensorSpec([None,num_kp,2],tf.float32),tf.TensorSpec([1,num_kp,2],tf.float32),tf.TensorSpec((),tf.bool),tf.TensorSpec((),tf.bool)])(self.__call__)
        super(ProcessKpDriving,self).__init__()
    
    def __call__(self,kp_driving,kp_source,relative,adapt_movement_scale):
        source_area = self.convex_hull_area(kp_source)
        driving_area = self.convex_hull_area(kp_driving)
        scales = tf.sqrt(source_area)/tf.sqrt(driving_area)
        ones = scales*0.0 + 1.0
        scales = tf.where((adapt_movement_scale & (ones>0)), scales, ones) #(10,)
        scales = tf.expand_dims(tf.expand_dims([scales],1),1)
        ones = tf.expand_dims(tf.expand_dims([ones],1),1)
        kp_value_diff = kp_driving - tf.expand_dims(kp_driving[0],0)
        kp_value_diff *= scales
        kp_new = kp_value_diff + kp_source
        kp_driving = tf.where((relative & (ones>0)),kp_new,kp_driving)[0]
        return kp_driving
    
    @tf.autograph.experimental.do_not_convert
    def convex_hull_area(self,X):
        num_kp = self.num_kp
        O = X * 0
        L = tf.shape(X)[0]
        n = tf.cast(num_kp,tf.int32)
        l = tf.cast(tf.argmin(X[:,:,0],1),tf.int32)
        p = l
        rng = tf.range(L)
        for i in range(num_kp):
            pt = tf.transpose(tf.stack([rng,p]),(1,0))
            ind = tf.expand_dims(tf.gather_nd(X,pt),1)
            O = tf.concat([O[:,0:i],ind,O[:,i+1:n]],1)
            q = (p + 1) % n
            q.set_shape(rng.shape)
            qt = tf.transpose(tf.stack([rng, q]),(1,0))
            for j in range(num_kp):
                p1 = (tf.gather_nd(X,qt)[:,1]-tf.gather_nd(X,pt)[:,1])
                p2 = (X[:,tf.cast(j,tf.int32),0] - tf.gather_nd(X,qt)[:,0])
                p3 = (tf.gather_nd(X,qt)[:,0]-tf.gather_nd(X,pt)[:,0])
                p4 = (X[:,tf.cast(j,tf.int32),1]-tf.gather_nd(X,qt)[:,1])
                bbase = p1 * p2 - p3 * p4
                b = bbase < 0
                q = tf.where(b,tf.repeat(tf.cast(j,tf.int32),L),q)
                q.set_shape(rng.shape)
                qt = tf.transpose(tf.stack([rng,q]),(1,0))
            p=tf.where((((q-l)<1) & ((q-l)>-1)), p, q)        
        j = tf.transpose(tf.concat([tf.expand_dims(O[:,1][:,1],1),O[:,1][:,2:],tf.expand_dims(O[:,1][:,0],1)],1),(1,0))
        k = tf.transpose(tf.concat([tf.expand_dims(O[:,0][:,1],1),O[:,0][:,2:],tf.expand_dims(O[:,0][:,0],1)],1),(1,0))
        inc = (tf.eye(L)*tf.tensordot(O[:,0],j,1))@tf.ones((L,1)) - (tf.eye(L)*tf.tensordot(O[:,1],k,1))@tf.ones((L,1))
        area = 0.5 * tf.math.abs(inc)[0]
        return area
    

def build_process_kp_driving(num_kp=10):
    return ProcessKpDriving(num_kp)
