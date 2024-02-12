# syntax=docker/dockerfile:1
FROM gxsu/mopmc-env
MAINTAINER gxsu
ARG no_threads=4
ENV PATH="/root/cmake/bin:${PATH}"
WORKDIR /root
COPY . .
WORKDIR /root/mopmc
RUN mkdir build
RUN ["/bin/bash", "configure.sh"]
RUN ["/bin/bash", "build.sh"]
ENTRYPOINT [ "/bin/bash"]