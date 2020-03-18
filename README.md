# DenoiSeg
Microscopy image analysis often requires the segmentation of objects, 
but training data for this task is typically scarce and hard to obtain. 
Here we propose DenoiSeg, a new method that can be trained end-to-end 
on only a few annotated ground truth segmentations. We achieve this by 
extending NoiseVoid, a self-supervised denoising scheme that can be 
trained on noisy images alone, to also predict dense 3-class segmentations. 
The reason for the success of our method is that segmentation can profit 
from denoising, especially when performed jointly within the same network. 
The network becomes a denoising expert by seeing all available raw data, 
while co-learning to segment, even if only a few segmentation labels are 
available. This hypothesis is additionally fueled by our observation that 
the best segmentation results on high quality (very low noise) raw data 
are obtained when moderate amounts of synthetic noise are added. This 
renders the denoising-task non-trivial and unleashes the desired co-learning 
effect. We believe that DenoiSeg offers a viable way to circumvent the 
tremendous hunger for high quality training data and effectively enables 
few-shot learning of dense segmentations.

## How to run
### Conda-Env
You can install the necessary packages into a conda-env with the `setup.py`
file by typing `pip install .` from within this cloned git-repo.

