# syntax=docker/dockerfile:1
FROM gxsu/mopmc-base-env
WORKDIR /root
COPY . .
RUN ["/bin/bash", "install-cmake.sh"]
ENV PATH="/root/cmake/bin:${PATH}"
ENTRYPOINT ["/bin/bash"]