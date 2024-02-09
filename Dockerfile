# syntax=docker/dockerfile:1
FROM gxsu/mopmc-test-env
# specify number of threads for parallel compilation
# by appending the following option to your build command:
# --build-arg no_threads=<value>
ARG no_threads=4
WORKDIR /root
COPY . .
RUN ["/bin/bash", "install-cmake.sh"]
ENTRYPOINT ["/bin/bash"]