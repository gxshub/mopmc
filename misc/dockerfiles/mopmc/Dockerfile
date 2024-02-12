# syntax=docker/dockerfile:1
FROM gxsu/mopmc-env
# specify number of threads for parallel compilation
# by appending the following option to your build command:
# --build-arg no_threads=<value>
ARG no_threads=4
ENV PATH="/root/cmake/bin:${PATH}"
WORKDIR /root
RUN git clone https://github.com/gxshub/mopmc.git
WORKDIR /root/mopmc
RUN mkdir build
RUN ["/bin/bash", "configure.sh"]
RUN ["/bin/bash", "build.sh"]
ENTRYPOINT [ "/bin/bash"]
