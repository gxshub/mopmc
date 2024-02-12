# syntax=docker/dockerfile:1
FROM gxsu/mopmc-env
MAINTAINER gxsu
ARG no_threads=4
ENV PATH="/root/cmake/bin:${PATH}"
WORKDIR /root/mopmc
COPY . .
RUN mkdir build
RUN sed -i -e 's/\r$//' ./configure.sh ./build.sh
RUN ["/bin/bash", "configure.sh"]
RUN ["/bin/bash", "build.sh"]
ENTRYPOINT [ "/bin/bash"]