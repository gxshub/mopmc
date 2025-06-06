# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
# specify number of threads for parallel compilation
# by appending the following option to your build command:
# --build-arg no_threads=<value>
ARG no_threads=4
WORKDIR /root
COPY . .
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev
RUN echo "--- Checking CMake Version ---" && cmake --version
WORKDIR /root
RUN git clone -b stable https://github.com/gxshub/carl-storm.git carl
#RUN git config --global user.email "you@example.com" && git config --global user.name "Your Name" && git clone https://github.com/moves-rwth/carl-storm.git carl
WORKDIR /root/carl/build
RUN cmake .. ; make lib_carl -j $no_threads
WORKDIR /root
#RUN git clone -b stable https://github.com/moves-rwth/storm.git
RUN git clone -b mopmc-dep https://github.com/gxshub/storm.git
WORKDIR /root/storm/build
RUN cmake .. ; make -j $no_threads
RUN apt-get -y install wget
ENV STORM_HOME=/root/storm
WORKDIR /root
COPY . .
RUN ["/bin/bash", "install-cmake.sh"]
ENV PATH="/root/cmake/bin:${PATH}"
ENTRYPOINT ["/bin/bash"]