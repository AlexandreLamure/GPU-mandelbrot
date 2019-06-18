# GPU-mandelbrot
This project is a **Mandelbrot fractal** implementation in C++ on **GPU** using **Nvidia CUDA**.

We wanted to benchmark our results, so we implemented **three versions** :
* CPU single-threaded version
* CPU multithreaded version
* GPU version

## Getting started
Two binaries are produced by our CMake.

### mandel
This one simply runs the algorithm and output a PNG file with the Mandelbrot fractal.
You can specify the version to run with the flag `-m` and one of the following: {`CPU`, `MT`, `GPU`}. 

### bench
This one benchmark the three versions using **Google Benchmark** library.


## Output
Here is an example of our Mandelbrot fractal:
![Mandelbrot fractal](readme/fractal.png?raw=true "Mandelbrot fractal")


## Results
We ran our benchmark on multiple image size and multiple fractal iteration number to get more detailed results:
![Image size benchmark](readme/size.png?raw=true "Image size benchmark")

![Iteration benchmark](readme/iterations.png?raw=true "Iteration benchmark")
