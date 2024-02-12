# syntax=docker/dockerfile:1
FROM gxsu/mopmc-base-env
WORKDIR /root
COPY . .
RUN ["/bin/bash", "install-cmake.sh"]
ENTRYPOINT ["/bin/bash"]