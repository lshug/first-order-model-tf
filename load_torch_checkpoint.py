import pickle
import sys
import numpy as np
import zipfile

mode = 'numpy'

arr_dict = {}
scount = 0

def _is_zipfile(f):
    read_bytes = []
    start = f.tell()

    byte = f.read(1)
    while byte != "":
        read_bytes.append(byte)
        if len(read_bytes) == 4:
            break
        byte = f.read(1)
    f.seek(start)
    
    local_header_magic_number = [b'P', b'K', b'\x03', b'\x04']
    return read_bytes == local_header_magic_number

def rebuild_as_numpy(storage, storage_offset, size, stride, *args, **kwargs):
    arr = np.reshape(storage.arr, size)
    arr = np.copy(np.lib.stride_tricks.as_strided(storage.arr, size, stride))
    if storage.preloaded:
        np.putmask(arr, np.ones(arr.shape, dtype='bool'), np.reshape(storage.arr, arr.shape))
    else:
        arr_dict[storage.ind] = arr
    return arr

class FloatStorage:
    def __init__(self, size):
        global scount
        self.arr = np.zeros(size, dtype='float32')
        self.dtype = 'float32'
        self.ind = scount
        self.preloaded = False
        scount += 1

class LongStorage:
    def __init__(self, size):
        global scount
        self.arr = np.zeros(size, dtype='int64')
        self.dtype = 'int64'
        self.ind = scount
        self.preloaded = False
        scount += 1

class InjectorUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if '_rebuild_tensor_v2' in name:
            return rebuild_as_numpy
        elif 'LongStorage' in name:
            return LongStorage
        elif 'FloatStorage' in name:
            return FloatStorage
        else:
            __import__(module)
            mod = sys.modules[module]
            klass = getattr(mod, name)
            return klass

def numpy_load(path):
    z = zipfile.ZipFile(path)
    loaded_storages = {}
    def load_tensor(data_type, size, key):
        name = f'archive/data/{key}'
        dtype = data_type(0).dtype
        storage = data_type(size)
        f = z.open(name)
        buffer = f.read()
        storage.arr = np.frombuffer(buffer, dtype=storage.dtype)
        storage.preloaded = True
        loaded_storages[key] = storage
    def persistent_load(saved_id):
        data = saved_id[1:]
        data_type, key, location, size = data
        obj = data_type(size)
        if key not in loaded_storages:
            load_tensor(data_type, size, key)
        storage = loaded_storages[key]
        return storage
    unpickler = InjectorUnpickler(z.open('archive/data.pkl'))
    unpickler.persistent_load = persistent_load
    return unpickler.load()

def numpy_load_legacy(path):
    deserialized_objects = {}
    def persistent_load(saved_id):
        data = saved_id[1:]
        data_type, root_key, location, size, metadata = data
        obj = data_type(size)
        if root_key not in deserialized_objects:
            deserialized_objects[root_key] = obj
        storage = deserialized_objects[root_key]
        return storage
    f = open(path, 'rb')
    for i in range(3):
        pickle.load(f)
    unpickler = InjectorUnpickler(f)
    unpickler.persistent_load = persistent_load
    result = unpickler.load()
    deserialized_storage_keys = pickle.load(f)
    offset = f.tell()
    for idx, key in enumerate(deserialized_storage_keys):
        arr = arr_dict[deserialized_objects[key].ind]
        buffer = f.read(8 + arr.nbytes)
        new_arr = np.zeros(arr.size, dtype=arr.dtype)
        new_arr.data.cast('B')[0:] = buffer[8:]
        new_arr = np.reshape(new_arr, arr.shape)
        np.putmask(arr, np.ones(arr.shape, dtype='bool'), new_arr)
    arr_dict.clear()
    return result 

def torch_load(path):
    import torch
    intermediate = torch.load(path, map_location=torch.device('cpu'))
    for k in intermediate['kp_detector'].keys():
        intermediate['kp_detector'][k] = intermediate['kp_detector'][k].numpy()
    for k in intermediate['generator'].keys():
        intermediate['generator'][k] = intermediate['generator'][k].numpy()
    return {'kp_detector':intermediate['kp_detector'], 'generator':intermediate['generator']}

def load_torch_checkpoint(path):
    if mode == 'numpy':
        f = open(path, 'rb')
        if _is_zipfile(f):
            f.close()
            return numpy_load(path)
        else:
            f.close()
            return numpy_load_legacy(path)
    else:
        return torch_load(path)
