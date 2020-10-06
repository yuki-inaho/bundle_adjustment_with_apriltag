#!/bin/bash -e

pip install -r requirements.txt
cd apriltag_detector_pywrapper
python setup.py install

#https://github.com/uoip/g2opy/issues/7
mkdir -p /app/g2opy/build && cd /app/g2opy/build
cmake .. && make -j8 && make install
#cd /app && rm -rf /app/g2opy/build