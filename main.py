
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

def update(val):
    
    gamma_v = gamma_slider.val
    gamma_v2 = gamma_slider2.val
    rot_v = rot_slider.val
    decay_v = decay_slider.val
    klim_v = kernel_lim_slider.val
    thres_v = thres_slider.val
    scaletint_v = scaletint_slider.val
    spacing_v = Spacing_slider.val

    
    ax1.imshow(main_image)
    ax1.set_title('Mask')
    ax1.axis('off')
    ax2.imshow(kernel_image_full)
    ax2.set_title('Original Kernel Image')
    ax2.axis('off')

    compshow = np.zeros_like(main_image)

    R_v = 0.8*scaletint_v
    G_v = 0.4*scaletint_v
    B_v = 0.1*scaletint_v
    


    comp = np.zeros(np.shape(main_image))
    size_const2 = (main_image.shape[0]/klim_v**1.5,main_image.shape[1]/klim_v**1.5)
    i = 0
    size_const1 = (0,0)
    while (size_const1 < size_const2):
        kernel_image_rot = ndimage.rotate(kernel_image_full,i*i*rot_v,reshape=False)
        kernel_image = np.array(ndimage.zoom(kernel_image_rot, (spacing_v*0.01*i**2, spacing_v*0.01*i**2, 1), order=0),dtype=np.float32)
        kernel_image[:,:,0] /= 1+i*R_v
        kernel_image[:,:,1] /= 1+i*G_v
        kernel_image[:,:,2] /= 1+i*B_v
        size_const1 = kernel_image.shape
        mask = (lum**gamma_v>thres_v)
        convolved_image = convolve_fft((main_image**gamma_v2)*mask, kernel_image)*(0.5*i**decay_v)
        comp += convolved_image
        compshow = np.sign(comp) * (np.abs(comp)) **(0.01+1/gamma_v2)

        ax1.imshow(mask.astype(np.float32))
        ax3.imshow(compshow/2+main_image)
        i += 1
 
        
        

    
    
    

    
 

main_image = np.array(Image.open('TestImage1.jpg'))
main_image = main_image / 255
main_image = ndimage.zoom(main_image, (.4, .4, 1), order=0)
lum = (0.299*main_image[:,:,0] + 0.587*main_image[:,:,0]  + 0.114*main_image[:,:,0] )
lum = np.stack([lum]*3, axis=-1)
kernel_image_full = np.array(Image.open('kernel7.jpg'))
plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
plt.subplots_adjust(bottom=0.50)
ax2.set_title('Move sliders to run (might take awhile)')



axTint = plt.axes([0.25, 0.3, 0.65, 0.03]) 
scaletint_slider = Slider(axTint, 'Tint', 0, 1, valinit=0.95, valstep=0.01)


axthres = plt.axes([0.25, 0.25, 0.65, 0.03]) 
thres_slider = Slider(axthres, 'Threshold', 0.0, 1, valinit=0.95, valstep=0.01)

axgamma = plt.axes([0.25, 0.2, 0.65, 0.03]) 
gamma_slider = Slider(axgamma, 'Gamma Mask', 0, 20, valinit=3, valstep=0.1)

axgamma2 = plt.axes([0.25, 0.35, 0.65, 0.03]) 
gamma_slider2 = Slider(axgamma2, 'Gamma Image', 0, 20, valinit=1, valstep=0.1)

axSpacing = plt.axes([0.25, 0.4, 0.65, 0.03]) 
Spacing_slider = Slider(axSpacing, 'Kernel scaling', 0.5, 4, valinit=20, valstep=0.1)



axrot= plt.axes([0.25, 0.15, 0.65, 0.03]) 
rot_slider = Slider(axrot, 'Rotation', 0, 20, valinit=0, valstep=0.1)

axdecay = plt.axes([0.25, 0.1, 0.65, 0.03]) 
decay_slider = Slider(axdecay, 'Scale->Bright', 0, 20, valinit=1, valstep=0.1)

axKernel_limit= plt.axes([0.25, 0.05, 0.65, 0.03]) 
kernel_lim_slider = Slider(axKernel_limit, 'Kernel size limit', 2, 15, valinit=2, valstep=0.1)

Spacing_slider.on_changed(update)
scaletint_slider.on_changed(update)
thres_slider.on_changed(update)
gamma_slider.on_changed(update)
gamma_slider2.on_changed(update)
rot_slider.on_changed(update)
decay_slider.on_changed(update)
kernel_lim_slider.on_changed(update)




plt.show()


