## Dockerfile Note: libxnvctrl0 Installation Workaround
This section of the Dockerfile manually handles the installation of the `libxnvctrl0` package.
```dockerfile
RUN apt-get update
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/./libxnvctrl0_575.57.08-0ubuntu1_amd64.deb -O /tmp/libxnvctrl0.deb
RUN dpkg -i /tmp/libxnvctrl0.deb || true
RUN apt-get update && apt-get install -y -f
RUN rm /tmp/libxnvctrl0.deb
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
```