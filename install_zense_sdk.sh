#!/bin/bash -e

mv /app/$PICOZENSE_LIB $PICOZENSE_INSTALL_DIR
chown -R root:root $PICOZENSE_INSTALL_DIR
cd $PICOZENSE_INSTALL_DIR

./install.sh

cat <<EOF > /usr/lib/pkgconfig/libpicozense.pc
prefix=$PICOZENSE_INSTALL_DIR
exec_prefix=\${prefix}
includedir=\${prefix}/Include
libdir=\${exec_prefix}/Lib
Name: libpicozense
Description: The Library for Pico Zense
Version: 1.0.0
Cflags: -I\${includedir}/
Libs: -L\${libdir} -lvzense_api -lImgPreProcess
EOF

ln -sf $(pkg-config --libs-only-L libpicozense | sed 's/^-L//')/* /usr/local/lib/
ldconfig