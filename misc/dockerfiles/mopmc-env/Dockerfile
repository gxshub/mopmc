# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
# specify number of threads for parallel compilation
# by appending the following option to your build command:
# --build-arg no_threads=<value>
ARG no_threads=4
RUN apt-get update && apt-get -y install wget
WORKDIR /root
RUN wget https://github.com/prismmodelchecker/prism/releases/download/v4.4/prism-4.4-linux64.tar.gz
RUN mkdir prism && \
    tar -xzf prism-4.4-linux64.tar.gz -C prism --strip-components=1
WORKDIR /root/prism
RUN ["/bin/bash", "install.sh"]
WORKDIR /root
RUN apt-get update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/./libxnvctrl0_575.57.08-0ubuntu1_amd64.deb -O /tmp/libxnvctrl0.deb
RUN dpkg -i /tmp/libxnvctrl0.deb || true
RUN apt-get update && apt-get install -y -f
RUN rm /tmp/libxnvctrl0.deb
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    libboost-all-dev \
    libcln-dev \
    libgmp-dev \
    libginac-dev \
    automake \
    libglpk-dev \
    libhwloc-dev \
    libz3-dev \
    libxerces-c-dev \
    libeigen3-dev && \
    rm -rf /var/lib/apt/lists/* # Clean up apt cache immediately to save space
RUN git clone -b stable https://github.com/gxshub/carl-storm.git carl
#RUN git config --global user.email "you@example.com" && git config --global user.name "Your Name" && git clone https://github.com/moves-rwth/carl-storm.git carl
WORKDIR /root/carl/build
RUN cmake .. ; make lib_carl -j $no_threads
WORKDIR /root
#RUN git clone -b stable https://github.com/moves-rwth/storm.git2
RUN git clone -b mopmc-dep https://github.com/gxshub/storm.git
WORKDIR /root/storm/build
RUN cmake .. ; make -j $no_threads
ENV PRISM_HOME=/root/prism
ENV STORM_HOME=/root/storm
WORKDIR /root
COPY . .
RUN ["/bin/bash", "install-cmake.sh"]
ENV PATH="/root/cmake/bin:${PATH}"
ENTRYPOINT ["/bin/bash"]