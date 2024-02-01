# MOPMC: A GPU-Accelerated Probabilistic Model Checking Tool for Multi-Objective Convex Queries

MOPMC is a multi-objective probabilistic model checking tool specialised for _convex queries_ on 
Markov Decision Processes (MDPs) with multiple objectives.
A convex query returns an (approximately) optimal point (and value) for a given convex function (viewed as a loss function) 
that is defined on the multi-dimensional objective space.
Examples of convex functions are Euclidean distance, MSE, variance, etc.
Currently, only total reward objectives are supported.


Built on top of [Storm](https://www.stormchecker.org)'s model parsing and building C++ API, 
MOPMC accepts a PRISM model format (for an MDP) and a PCTL/LTL-style property specification.
One important feature of MOPMC is the utilisation of GPU hardware acceleration for valuation-iteration computing.
The convex queries in MOPMC can scale to a large number of objectives.

For benchmarking, MOPMC also implements the achievability queries, which are supported by other existing probabilistic model checking tools.


## Getting Started

### Built from Source

This build is known to work on Ubuntu 20.04 LTS.

Before starting, install Storm and its _dependencies_ from source code. See the Storm [documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html) for the installation procedure.
This project is built with CMake (which is included in Storm's dependencies).

<!-- This project uses cmake which should be bundled with Ninja. If Ninja is available you will be able
to make use of the convenient configurations and build script.-->

Installation of the CUDA Toolkit 12.0 (or above) is required.
This version is essential as it provides 64bit numeric types for the GPU and provides more modern
sparse matrix multiplication algorithms from NVIDIA CuSparse.
See the [CUDA installation documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) for detailed information.
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
### Use Pre-configured Docker Image
A pre-configured environment for compiling MOMPC is defined in a [__mopmc-env__](https://hub.docker.com/r/gxsu/mopmc-env) 
Docker image, which is available in the Docker Hub.
To run a Docker container with GPU acceleration, the 
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) is required.
Follow the 
[installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to install the toolkit and configure Docker.

The following procedure assumes that the host system is a Linux system, but in principle the Docker image can be deployed in all OS with the NVIDIA Driver,
NVIDIA Container Toolkit and Docker being installed.

Pull the image:
```shell
docker pull gxsu/mopmc-env
```
Clone this project and run the Docker container as follows:
```shell
export hostdir=<path to this project's directory> ;
export sharedir=/root/mopmc ;
docker run --mount type=bind,source=$hostdir,target=$sharedir --rm -it --runtime=nvidia --gpus all gxsu/mopmc-env
```
Then, compile MOPMC:
```shell
cd ~ ; mkdir build ; ./configure.sh ; ./build.sh
```

### Running MOPMC
To run a convex query:
```shell
./build/mopmc -m examples/dive_and_rise/dive_and_rise.nm -p examples/dive_and_rise/dive_and_rise_prop_100.props -q convex 
```
To run an achievability query:
```shell
./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability
```
To run an achievability query in Storm:
```shell
$STORM_HOME/build/bin/storm --prism examples/multiobj_scheduler05.nm --prop examples/multiobj_scheduler05.pctl
```

## About Model and Property Specification
MOPMC accepts the standard PRISM model format for MDPs. For property specification, 
it accepts the PCTL/LTL-style multi-objective achievability properties, 
which are adopted by existing PMC tools such as Storm and PRISM. 
For convex queries, it interprets an achievability property in the following way: 
Suppose a property specification is given as
```multi(R{"time"}<=14.0 [ F "tasks_complete" ], R{"energy"}<=1.25 [  F "tasks_complete" ])```
for the MDP in `multiobj_scheduler05.nm`, and the loss function is MSE.
Let $x_t$ and $x_e$ denote the total rewards for `"time"` and `"energy"`, respectively.
The values $x_t$ and $x_e$ are subject to the computed scheduler for the MDP.
A convex query returns $x_t$ and $x_e$ that minimise $((x_t-14.0)^2 + (x_e-1.25)^2)\cdot 0.5$. 
