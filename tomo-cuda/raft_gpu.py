import numpy as np
import time
import ctypes
from ctypes import *
import pyraft


libraft = ctypes.CDLL('./libraft.so', mode=ctypes.RTLD_GLOBAL)
soma1 = libraft.raft_backprojection_slantstack_gpu
soma1.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t, c_size_t]
back = libraft.Back
back.argtypes = [POINTER(c_float), POINTER(c_float), c_size_t, c_size_t, c_size_t]   

img_size = 2048
rays = 2048
angles = 1000

gear = pyraft.gear([img_size,img_size])
gear_size = gear.shape[0]*gear.shape[1]

sino  = pyraft.radon(gear,[rays, angles])
sino_size = sino.shape[0]*sino.shape[1]


print img_size, rays, angles, gear_size, sino_size



#allocate matrix
a =  np.frombuffer(sino).astype('float32')
b = np.zeros(gear_size).astype('float32')
#get matrix pointers as CTYPES
a_p = a.ctypes.data_as(POINTER(c_float))
b_p = b.ctypes.data_as(POINTER(c_float))


t = time.time()
back(b_p, a_p, img_size, rays, angles)
elapsed = time.time() - t

print elapsed

img_ini = a.reshape(sino.shape)
img_final = b.reshape(gear.shape)

pyraft.imagesc(img_ini, img_final)

