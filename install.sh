#!/bin/bash -e

pip install -r requirements.txt
cd apriltag_detector_pywrapper
python setup.py install

mkdir -p /app/g2opy/build && cd /app/g2opy/build
cmake .. && make -j8 && make install
cd .. && python setup.py install

mkdir -p /app/pangolin/build && cd /app/pangolin/build
cmake .. && make -j8
cd .. && python setup.py install
