# MOPMC: A GPU-Accelerated Probabilistic Model Checking Tool for Multi-Objective Convex Queries

MOPMC is a multi-objective probabilistic model checking tool specialised for _convex queries_ on
Markov Decision Processes (MDPs) with multiple objectives.
A convex query returns an (approximately) optimal point (and value) for a given convex function (viewed as a loss function)
that is defined on the multi-dimensional objective space.
Examples of convex functions are MSE, variance, etc.
Currently, only total reward objectives are supported.


Built on top of [Storm](https://www.stormchecker.org)'s model parsing and building C++ API,
MOPMC accepts a PRISM model format (for an MDP) and a PCTL/LTL-style property specification.
One important feature of MOPMC is the utilisation of GPU hardware acceleration for valuation-iteration computing.
The convex queries in MOPMC can scale to a large number of objectives.

For benchmarking, MOPMC also implements the achievability queries, which are supported by other existing probabilistic model checking tools.


## Getting Started

### Built from Source

#### Ubuntu 20.04 LTS
This build is known to work on Ubuntu 20.04 LTS

#### Storm
MOMPC utilises a re-build of [**Storm stable v1.8.1**](https://github.com/moves-rwth/storm/tree/3f74f3e59acfba3b61c686af01a864962d44af97) as a library for parsing,
model building and processing.
A slight modification is make to Storm's source code to support model export in MOPMC.
The modified version is included in this [fork](https://github.com/gxshub/storm/tree/mopmc-dep), which extends Storm stable v1.8.1.

The modified storm can be built using **CMake 3.16.3**. Note that other version of CMake may be incompatible with Storm stable v1.8.1.

To build storm as a library, run `sudo make install -j <num_of_threads>` rather than `make`.
This way, this project can use `find_package(storm)` in `CMakeLists.txt` to load Storm.


#### CUDA Toolkit

This project uses [`FindCUDAToolkit`](https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html) CMake script, which is available version in CMake 3.17+, to identify the location of CUDA.

Installation of the CUDA Toolkit 12.0 (or above) is required (see the
[NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)).
This version is essential as it provides 64bit numeric types for the GPU and provides more modern
sparse matrix multiplication algorithms from NVIDIA CuSparse.
Use `nvcc --version` and `nvidia-smi` to check the installed toolkit and driver versions, respectively. Also note that the [version compactibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility) between the CUDA Toolkit and the NVIDIA Driver.
<!--
```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06             Driver Version: 525.125.06   CUDA Version: 12.2     |
+-----------------------------------------+----------------------+----------------------+
```
-->
After installation, append the toolkit to `PATH`, e.g., by adding the following line to either `.bashrc` or `.profile`:
```shell
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
```

Additionally, if an IDE, such as CLion, is used, then also set the `LD_LIBRARY_PATH` to contain the toolkit's lib64 directory. This can be done by adding the following line into to  either `.bashrc` or `.profile`:
```shell
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
This avoids errors by the IDE debug compiler relating to setting `CMAKE_CUDA_ARCHITECTURES`.

The process of compiling MOPMC from the source is as follows:
Clone this project, `cd` into the project root, and execute

```shell
mkdir build ; ./configure.sh ; ./build.sh
```
To test the	 build is working, run the executable using the convenience script:
```shell
./test-run.sh
```

#### CMake Versions (!)
Storm v1.8.1 uses **CMake 3.16.3**. MOPMC can be built using CMake **3.28.0** (and requires at least **3.22**).
Therefore, [different CMake versions](https://cmake.org/download/) are recommended.

### Use Pre-configured Docker Image
A pre-configured environment for compiling MOMPC is defined in a [__mopmc-env__](https://hub.docker.com/r/gxsu/mopmc-env)
Docker image, which is in the Docker Hub.
This Docker image contains a Ubuntu 20.04 OS and is built to support __AMD64 (x86_64)__ and __ARM64 (Apple silicon)__ architectures.
It has been tested in the host OSs __Ubuntu 20.04 LTS__, __Windows 10__ and __MacOS__ ___with or without___ NVIDIA GPU
(In the latter case, the command of running MOPMC must include the option `-v standard` (see below)).

To run a Docker container with GPU acceleration, the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) is required.
Follow the
[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to install the toolkit and configure Docker.

Clone this project, and build a Docker image:
```shell
docker build --tag mopmc .
```
Then, run the image:
```shell
docker run --rm -it --runtime=nvidia --gpus all mopmc
```
If NVIDIA GPU is not used, run it as follows:
```shell
docker run --rm -it mopmc
```
<!---
Pull the image:
```shell
docker pull gxsu/mopmc-env
```
Clone this project and run the Docker container as follows:
```shell
export hostdir=<path to this project's directory>
export sharedir=/root/mopmc
docker run --mount type=bind,source=$hostdir,target=$sharedir --rm -it --runtime=nvidia --gpus all gxsu/mopmc-env
```
If GPU acceleration is not used, the last command should be: 
```shell
docker run --mount type=bind,source=$hostdir,target=$sharedir --rm -it gxsu/mopmc-env
```
Then, in the Docker container, build MOPMC as follows:
```shell
cd ~ ; mkdir build ; ./configure.sh ; ./build.sh
```
If the host OS is Windows, use the following command to convert some characters before building MOPMC:
```shell
sed -i -e 's/\r$//' ./configure.sh ./build.sh
```
-->

### Use Docker Image with Pre-built MOPMC
An [__MOPMC Docker image__](https://hub.docker.com/repository/docker/gxsu/mopmc/general)
with a ready-to-run MOPMC build is available in the Docker Hub.
__As this project is being actively developed, the pre-built version may not be the latest.__

(Optionally) pull the image with a pre-built MOPMC:
```shell
docker pull gxsu/mopmc
```
Run the image:
```shell
docker run --rm -it --runtime=nvidia --gpus all gxsu/mopmc
```
with NVIDIA GPU (note that [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) must be [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)), or
```shell
docker run --rm -it gxsu/mopmc
```
without NVIDIA GPU.


### Convex Query in MOPMC
To run a convex query:
```shell
./build/mopmc -m examples/dive_and_rise/dive_and_rise_action_rewards.nm -p examples/dive_and_rise/dive_and_rise_prop_cq_10.props -q convex 
```
To run a query without GPU acceleration for value iteration:
```shell
./build/mopmc -m examples/dive_and_rise/dive_and_rise_action_rewards.nm -p examples/dive_and_rise/dive_and_rise_prop_cq_10.props -q convex -v standard
```
To see all the running options:
```shell
./build/mopmc -h
```

To export model only:
```shell
./build/mopmc -m examples/dive_and_rise/dive_and_rise_action_rewards.nm \
-p examples/dive_and_rise/dive_and_rise_prop_cq_10.props \
-e out/dive_and_rise
```

To run a query and export the returned schedulers:
```shell
./build/mopmc -m examples/dive_and_rise/dive_and_rise_action_rewards.nm -p \
examples/dive_and_rise/dive_and_rise_prop_cq_10.props -q convex  \
-x out/dive_and_rise
```

### MOPMC and Other Tools (Achievability Query)


To run an achievability query in MPMC:
```shell
./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability
```

To run an achievability query in Storm:
```shell
$STORM_HOME/build/bin/storm --prism examples/multiobj_scheduler05.nm --prop examples/multiobj_scheduler05.pctl
```

To run an achievability query in PRISM:
```shell
$PRRIM_HOME/bin/storm examples/multiobj_scheduler05.nm examples/multiobj_scheduler05.pctl
```

### About Model and Property Specification
MOPMC accepts the standard PRISM model format for MDPs. For property specification,
it accepts the PCTL/LTL-style multi-objective achievability properties,
which are adopted by existing PMC tools such as Storm and PRISM.
For convex queries, it interprets an achievability property in the following way:
Suppose a property specification is given as
```multi(R{"time"}<=14.0 [ F "tasks_complete" ], R{"energy"}<=1.25 [  F "tasks_complete" ])```
for the MDP in `multiobj_scheduler05.nm`, and the loss function is MSE.
Let $x_t$ and $x_e$ denote the total rewards for `"time"` and `"energy"`, respectively.
The values $x_t$ and $x_e$ are subject to the computed scheduler for the MDP.
A convex query returns $x_t$ and $x_e$ that minimise the objective $(x_t^2 + x_e^2)/2$ subject to $x_t\leq 14$ and $x_t\leq 1.25$.
If the loss function is the variance, then the objective is
that minimise $((x_t - \overline{x})^2 + (x_e - \overline{x})^2)/2$ where $\overline{x}= (x_t+x_e)/2$.