# Noise2Seg
Recent works have shown how image denoising prior to segmentation can enable better segmentation, 
especially in the presence of low amount of annotated training data for training a segmentation 
network. In this work, we handle both denosing and segmentation in a joint framework rather than 
using denoising as just a preprocessing block. We show that the synergy between denoising and 
segmentatioon taks can be harvested better when treating them together. To this end, we propose 
a novel end-to-end deep learning based joint training scheme. Our framework is based on 
unsupervised denoising scheme called Noise2Void which can use all available noisy data for
training a neural network. We extend Noise2Void denoising loss with a 3-class segmentation loss 
for jointly training a deep learning network in an end-to-end manner. We show that this enables 
the joint network to generalize better for segmetation tasks and outperforms all other methods 
which treat denoising and segmentation as two step problem. We also investigate the relative 
importance of denoising and segmentation parts during the joint training and report the best 
schedules for weighting these two tasks during joint learning. Our results show that our proposed 
joint training can greatly benefit segmentation across different noise regimes, especially when 
only little annotated training data is available. We belive our approach enables efficient 
training of deep learning based segmentation networks by addressing the central bottleneck of 
needing huge amounts of annotated data during training.

## How to run
### Conda-Env
You can either install the necessary packages into a conda-env with the setup.py file by typing `pip install .` 
from within this cloned git-repo or you use a singularity container.

### Build Singularity Container
1. Install Singularity
2. `sudo singularity build noise2seg.sim singularity/noise2seg.Singularity`

### Run Singularity Container
Now you can run a jupyter-server with this container:
`singularity run --nv -B singularity/user:/run/user -B ./:/notebooks singularity/noise2seg.simg`

### Run Singularity Headless
1. Install PyInquirer
2. `python3 run_noise2seg.py`

For the headless to work you have to set following parameters in the file `run_noise2seg.py`:
```
singularity_path = "/projects/Noise2Seg/singularity/"
base_dir = "/projects/Noise2Seg"
gitrepo_path = join(base_dir, 'Noise2Seg')
base_path_data = join(base_dir, "data")
base_path_exp = join(base_dir, "experiments")
```
