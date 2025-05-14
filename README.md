# FFT-Based-Bloom
FFT-Based Bloom Effect using GPU Acceleration

This project implements a generalized bloom post-processing effect using FFT-based convolution, with GPU acceleration for real-time performance. Bloom, often used in computer graphics and digital art, simulates the glow that appears around bright light sources, enhancing realism and visual appeal. Unlike simple blur-based bloom implementations, this version supports custom kernel shapes, multi-scale convolution, and fine-tuning controls to simulate physically inspired light diffraction patterns.

## Project Highlights
Customizable Bloom Kernel: Users can draw their own kernel to control the glow shape.

FFT Convolution: Leverages the Fast Fourier Transform for efficient large-kernel convolutions.

GPU Acceleration: Implements pyVkFFT, an OpenCL-based FFT library, reducing bloom rendering time from over 1 second to under 0.005 seconds on compatible GPUs.

Brightness Masking: Supports threshold-based and gamma-based brightness selection to isolate glowing regions.

Interactive Controls: GUI with sliders to adjust brightness threshold, gamma correction, kernel scaling, and blend intensity.

## How It Works
Brightness Filtering: Bright parts of the image are selected based on a dynamic threshold or nonlinear gamma masking.

Custom Kernel Scaling: A user-defined kernel is resized and padded to match image dimensions, enabling multiscale glow simulation.

FFT-Based Convolution: Each color channel of the masked image is convolved with the scaled kernels using FFT in the frequency domain.

Combining Results: The filtered images are summed and blended with the original to produce the final bloomed image.

This approach follows principles used in advanced game engines (e.g., Unreal Engine), but is implemented in Python using general-purpose GPU computing.

## Performance
Implementation	FFT Library	Average Runtime
CPU (Scipy)	scipy.fft	~1.115 seconds
GPU (OpenCL)	pyVkFFT	~0.003 seconds

The GPU implementation provides a 400x speedup over the CPU version, demonstrating the advantage of parallelized FFT computation.

## Visual Output
The bloom behaves as expected in various test cases, with realistic glow shapes around bright regions. Multiple kernel sizes and shapes were tested, showcasing the flexibility of the approach.

<p align="center"><i>(Example images omitted here—see the "Results" section in the project report or screenshots folder (will add later).)</i></p>

## Dependencies
Python 3.8+

numpy

pyvkfft

Pillow

matplotlib (for GUI sliders)

OpenCL compatible GPU
