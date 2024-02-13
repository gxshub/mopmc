# syntax=docker/dockerfile:1
FROM gxsu/mopmc-env
MAINTAINER gxsu
WORKDIR /root/mopmc
COPY . .
RUN mkdir build
RUN sed -i -e 's/\r$//' ./configure.sh ./build.sh ./test-run.sh
RUN ["/bin/bash", "configure.sh"]
RUN ["/bin/bash", "build.sh"]
ENTRYPOINT [ "/bin/bash"]
