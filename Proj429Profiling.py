
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, ndimage,signal
from PIL import Image
import torch
from matplotlib.widgets import Slider
import time
import pyvkfft
import pyopencl as cl
import pyopencl.array as cla
import pyvkfft.opencl
from pyvkfft.fft import fftn as vkfftn, ifftn as vkifftn, rfftn as vkrfftn, irfftn as vkirfftn
from pyvkfft.opencl import VkFFTApp
from pyvkfft.accuracy import l2, li
import os
import cProfile
import pstats
import io
from IPython import display





if 'PYOPENCL_CTX' in os.environ:
    ctx = cl.create_some_context()
else:
    ctx = None
    for p in cl.get_platforms():
        for d in p.get_devices():
            if d.type & cl.device_type.GPU == 0:
                continue
            print("Selected device: ", d.name)
            ctx = cl.Context(devices=(d,))
            break
        if ctx is not None:
            break
cq = cl.CommandQueue(ctx)


def convolve_fft(image, kernel):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=2)
    if kernel.ndim == 2:
        kernel = np.expand_dims(kernel, axis=2)
    
    result = np.zeros_like(image, dtype=float)
    print(image.shape)
    for i in range(image.shape[2]):
        result[:,:,i] = convolve_fft_single_channel(image[:,:,i], kernel[:,:,i])
    result = (result - result.min()) / (0.1 + result.max() - result.min())
    return result

def convolve_fft_single_channel(image, kernel):
    image = image.astype(np.complex64)
    kernel = kernel.astype(np.complex64)
    
    padded_kernel = np.zeros_like(image)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    
    padded_kernel = np.roll(padded_kernel, (-kh // 2, -kw // 2), axis=(0, 1))
     
    fft_image =  vkfftn(cla.to_device(cq,image))
    fft_kernel =  vkfftn(cla.to_device(cq,padded_kernel))
    fft_result = fft_image * fft_kernel
    result = (vkifftn(fft_result).get()).real
    
    return result


def convolve_fft_single_channel_NUMPYVERSION(image, kernel):
    image = image.astype(float)
    kernel = kernel.astype(float)
    
    padded_kernel = np.zeros_like(image)
    kh, kw = kernel.shape
    padded_kernel[:kh, :kw] = kernel
    
    padded_kernel = np.roll(padded_kernel, (-kh // 2, -kw // 2), axis=(0, 1))
    
    fft_image = fftpack.fft2(image)
    fft_kernel = fftpack.fft2(padded_kernel)
    
    fft_result = fft_image * fft_kernel
    
    result = fftpack.ifft2(fft_result).real
    
    return result




main_image = np.array(Image.open('TestImage1.jpg'))
main_image = main_image / 255
main_image = ndimage.zoom(main_image, (0.3, 0.3, 1), order=0)
lum = (0.299*main_image[:,:,0] + 0.587*main_image[:,:,0]  + 0.114*main_image[:,:,0] )
lum = np.stack([lum]*3, axis=-1)
kernel_image_full = np.array(Image.open('kernel7.jpg'))




gamma_v = 1
gamma_v2 = 1
rot_v = 0
decay_v = 1
klim_v = 2
thres_v = 1
scaletint_v = 1
spacing_v = 1




comp = np.zeros(np.shape(main_image))
size_const2 = (main_image.shape[0]/klim_v**1.5,main_image.shape[1]/klim_v**1.5)
size_const1 = (0,0)
pr = cProfile.Profile()
pr.enable()
i = 0
while (size_const1 < size_const2):
    kernel_image_rot = ndimage.rotate(kernel_image_full,i*i*rot_v,reshape=False)
    kernel_image = np.array(ndimage.zoom(kernel_image_rot, (spacing_v*0.01*i**2, spacing_v*0.01*i**2, 1), order=0),dtype=np.float32)
    size_const1 = kernel_image.shape
    mask = (lum**gamma_v>thres_v)
    convolved_image = convolve_fft((main_image**gamma_v2)*mask, kernel_image)*(2*i**decay_v)
    comp += convolved_image
    compshow = np.sign(comp) * (np.abs(comp)) **(0.01+1/gamma_v2)
    i += 1
    

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
    f.write(s.getvalue())