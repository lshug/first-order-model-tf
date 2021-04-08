import pickle
import sys
import numpy as np
import struct

arr_dict = {}
def rebuild_as_numpy(storage, storage_offset, size, stride, *args, **kwargs):
    arr = np.copy(np.lib.stride_tricks.as_strided(storage.arr, size, stride))
    #arr = np.reshape(storage.arr, size)
    arr_dict[storage.ind] = arr
    return arr

scount = 0

class FloatStorage:
    def __init__(self, size):
        global scount
        self.arr = np.zeros(size, dtype='float32')
        self.dtype = 'float32'
        self.ind = scount
        scount += 1

lscount = 0
class LongStorage:
    def __init__(self, size):
        global scount
        self.arr = np.zeros(size, dtype='long')
        self.dtype = 'long'
        self.ind = scount
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

def load_torch_checkpoint(path):
    deserialized_objects = {}
    def persistent_load(saved_id):
        data = saved_id[1:]
        data_type, root_key, location, size, view_metadata = data
        if root_key not in deserialized_objects:
            obj = data_type(size)
            obj._torch_load_uninitialized = True
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
        nbytes = len(buffer[8:])
        new_arr = np.zeros(arr.size, dtype=arr.dtype)
        new_arr.data.cast('B')[0:] = buffer[8:]
        new_arr = np.reshape(new_arr, arr.shape)
        np.putmask(arr, [np.ones(arr.shape, dtype='bool')], new_arr)
    arr_dict.clear()
    return result 
