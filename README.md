# MOPMC: A GPU-Accelerated Probabilistic Model Checking Tool for Multi-Objective Convex Queries

MOPMC is a multi-objective probabilistic model checking tool specialised for _convex queries_ on Markov Decision Processes (MDPs) with multiple objectives.
A convex query returns an (approximately) optimal point (and value) for a given convex function (viewed as a loss function) that is defined on the objective space.
Examples of convex functions are Euclidean distance, MSE, variance, etc.
Currently, only total reward objectives are supported.


Built atop [Storm](https://www.stormchecker.org)'s C++ API of model parsing and building, MOPMC accepts a PRISM model file (an MDP) and a PCTL/LTL property specification.
One key feature of MOPMC is the utilisation of GPU hardware acceleration for valuation-iteration computing.
The convex queries can scale to a large number of objectives.

For benchmarking, MOPMC also implements the acheivability queries, which are supported by other existing probabilistic model checking tools.


## Getting Started

### Built from Source

This build is known to work on Ubuntu 20.04 LTS.

Before starting, install Storm and its _dependencies_ from source code. See the Storm [documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html) for the installation procedure.
This project is built with CMake (which is included in Storm's dependences).

<!-- This project uses cmake which should be bundled with Ninja. If Ninja is available you will be able
to make use of the convenient configurations and build script.-->

Installation of The CUDA Toolkit 12.xx (or above) is required.
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

```bash
export PATH=/usr/local/cuda-12.2/bin${PATH:+:${PATH}}
```

Additionally, if an IDE, such as CLion, is used, then also set the `LD_LIBRARY_PATH` to contain the toolkit's lib64 directory. This can be done by adding the following line into to  either `.bashrc` or `.profile`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
This avoids errors by the IDE debug compiler relating to setting `CMAKE_CUDA_ARCHITECTURES`.

<!--
If your IDE cannot find the Storm header files, you can specify the header search paths so that the Storm source directories
can be indexed (see [Manage CMake project files](https://www.jetbrains.com/help/clion/managing-cmake-project-files.html#nonprj_files)).
This can be done by adding the following line into the current [`CMakeList.txt`](./CMakeLists.txt) file:
```cmake
set(storm_INCLUDE_DIR, ./storm)
```
where `storm` is a symlink to `<YOUR_STORM_ROOT_DIRECTORY>/build/src/storm` created in the project root.
-->

The process of compiling MOPMC from the source is as follows:
Clone this project, `cd` into the project root, and execute

```
mkdir build ; ./configure.sh ; ./build.sh
```
To test the	 build is working, run the executable using the convenience script:
```bash
./test-run.sh
```
### Use Docker Container
A pre-configured environment for runing MOMPC is defined in a [__mopmc-env__](https://hub.docker.com/r/gxsu/mopmc-env) Docker image in the Docker Hub.
To run a Docker container with GPU acceleration, the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) is required (c.f., the toolkit's [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
Install docker and run
```bash
docker pull gxsu/mopmc-env
```
Close this project and run
```bash
export hostdir=<the folder of this project> ;
export sharedir=/root/mopmc ;
docker run --mount type=bind,source=$hostdir,target=$sharedir --rm -it --runtime=nvidia --gpus all gxsu/mopmc-env
```
Then, compile MOPMC:
```bash
cd ~ ; mkdir build ; ./configure.sh ; ./build.sh
```

### Running MOPMC
To run a convex query:
```bash
./build/mopmc -m examples/dive_and_rise/dive_and_rise.nm -p examples/dive_and_rise/dive_and_rise_prop_100.props -q convex 
```

To run an achievability query:
```bash
./build/mopmc -m examples/multiobj_scheduler05.nm -p examples/multiobj_scheduler05.pctl -q achievability
```

<!-- This project only computes multi-objective model checking of convex queries. -->

<!--
## Development

`src/main.cpp` is the entry point of the project. 

The first call is to `mopmc::check` which parses a model as a Prism model along with 
properties from a `.pctl` file. These are argument inputs with the first being model and the
second being property inputs. 

Model parsing is done using Storm parsing methods and once done multi-objective model
checking is done by calling:
```c++
mopmc::multiobjective::performMultiObjectiveModelChecking(env, *mdp, formulas[0]->asMultiObjectiveFormula());
```

This class method first preprocesses the multi-objective formulas and model by calling 
methods in 
```c++
src/mopmc-src/model-checking/MultiObjectivePreprocessor.cpp(h)
```

After model construction is complete, MOPMC model checking is conducted using
the methods and classes in `src/mopmc-src/model-checking/MOPMCModelChecking.cpp(h)`.
The class often makes reference to the solvers both `c++` and `CUDA` based located in
`src/mopmc-src/solvers`.
-->
