# syntax=docker/dockerfile:1
FROM gxsu/mopmc-env
ENV PATH="/root/cmake/bin:${PATH}"
WORKDIR /root
RUN git clone https://github.com/gxshub/mopmc.git
WORKDIR /root/mopmc
RUN mkdir build
RUN ["/bin/bash", "configure.sh"]
RUN ["/bin/bash", "build.sh"]
ENTRYPOINT [ "/bin/bash"]
