#!/bin/bash
export ARCH=$(dpkg --print-architecture)
if [[ $ARCH == "amd64" ]]
then
  echo $ARCH >> /root/arch-info ; \
  wget https://github.com/Kitware/CMake/releases/download/v3.28.2/cmake-3.28.2-linux-x86_64.tar.gz ; \
  tar -xvzf /root/cmake-3.28.2-linux-x86_64.tar.gz ; \
  mv /root/cmake-3.28.2-linux-x86_64 /root/cmake ; \
  rm /root/cmake-3.28.2-linux-x86_64.tar.gz
elif [[ $ARCH == "arm64" ]]
then
  echo $ARCH >> /root/arch-info ; \
  wget https://github.com/Kitware/CMake/releases/download/v3.28.2/cmake-3.28.2-linux-aarch64.tar.gz ; \
  tar -xvzf /root/cmake-3.28.2-linux-aarch64.tar.gz ; \
  mv /root/cmake-3.28.2-linux-aarch64 /root/cmake ; \
  rm /root/cmake-3.28.2-linux-aarch64.tar.gz
else
  echo "unsupported arch" >> /root/arch-info
fi
# echo 'export PATH=/root/cmake/bin${PATH:+:${PATH}}' >>~/.bashrc