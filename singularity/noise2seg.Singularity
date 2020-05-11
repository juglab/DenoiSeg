BootStrap: docker
# This image is built on the tensorflow 1.15 gpu docker image. 
# This image comes with CUDA-9.0 and all required tf dependencies.
From: tensorflow/tensorflow:1.15.0-gpu-py3
	
%post
    apt-get -y update
	

    # Install required Python packages
    pip install n2v
    pip install numpy
    pip install scipy
    pip install matplotlib
    pip install six
    pip install keras==2.2.5
    pip install tifffile
    pip install tqdm
    pip install csbdeep==0.4.1
    pip install numba
    pip install scikit-learn
    pip install scikit-image
    pip install jupyter

    apt-get autoremove -y
    apt-get clean

	
    # Remove the example notebooks
    rm -rf /notebooks/*
	
    # Make data directory
    mkdir /data
	
%runscript
    echo "Starting notebook..."
    echo "Open browser to localhost:8888"
	exec jupyter notebook --port=8888 --no-browser --ip="0.0.0.0" --notebook-dir='/notebooks'
