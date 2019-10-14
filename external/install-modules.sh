#!/bin/sh

cd eigen-eigen-07105f7124f9
mkdir build
cd build
cmake ..
sudo make install 
cd ../..
cd Mesh_Reconstruction
mkdir build
cd build
cmake ..
sudo make install
cd ../..
cd Curvox
sudo python setup.py install
cd ..
cd pycollada 
sudo python setup.py install
cd ..
cd python-pcl
sudo python setup.py install
cd ..
cd PyMCubes
sudo python setup.py install
cd ..
cd  binvox-rw-py
sudo python setup.py install

