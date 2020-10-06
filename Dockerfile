
FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN sed -i -r 's|(archive\|security)\.ubuntu\.com/|ftp.jaist.ac.jp/pub/Linux/|' /etc/apt/sources.list && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y build-essential apt-utils ca-certificates \
    cmake git pkg-config software-properties-common \
    libswscale-dev wget autoconf automake unzip curl libeigen3-dev \
    python-dev python-pip libavcodec-dev libavformat-dev libgtk2.0-dev libv4l-dev usbutils \
    libhdf5-100 libhdf5-cpp-100 hdf5-tools hdf5-helpers libhdf5-dev libhdf5-doc \
    python3-tk python3-pip tk-dev libv4l-dev libqglviewer-dev-qt4 libsuitesparse-dev libglew-dev && \
    # SDK dependency
    add-apt-repository ppa:nilarimogard/webupd8 && \
    apt-get update && apt-get install -y libvdpau-va-gl1 i965-va-driver vdpauinfo libvdpau-dev python-tk && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python & \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip && \
    pip install cython==0.29.20 numpy==1.16.0

WORKDIR /app
ENV OPENCV_VERSION="3.4.10"
RUN mkdir -p /app/opencv-$OPENCV_VERSION/build
RUN curl -L https://github.com/opencv/opencv/archive/$OPENCV_VERSION.tar.gz | tar xz

WORKDIR /app/opencv-$OPENCV_VERSION/build
RUN cmake -DWITH_TBB=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DPYTHON_EXECUTABLE=$(which python) \
    -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -DWITH_V4L=ON \
    -DWITH_EIGEN=ON \
    -DEIGEN_INCLUDE_PATH=/usr/include/eigen3 \
    -DOPENCV_GENERATE_PKGCONFIG=YES \
     ..

RUN make -j8 install && make clean && ldconfig

WORKDIR /app
COPY . /app

RUN ./install.sh

CMD [ /bin/bash ]