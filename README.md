# DenoiSeg
Microscopy image analysis often requires the segmentation of objects, but training data for such a task is hard to obtain.
Here we propose DenoiSeg, a new method that can be trained end-to-end on only a few annotated ground truth segmentations. 
We achieve this by extending NoiseVoid, a self-supervised denoising scheme that can be trained on noisy images, to also predict dense 3-class segmentations. 
The reason for the success of our method is that segmentation can profit from denoising especially when performed within the same network jointly.
The network becomes a denoising expert by seeing all available raw data, while  co-learning to segment even if only a few segmentation labels are available.
This hypothesis is additionally fueled by our observation that the best segmentation results on high quality (virtually noise free) raw data are performed when moderate amounts of synthetic noise are added. 
This renders the denoising-task non-trivial and unleashes the co-learning effect.
We believe that DenoiSeg offers a viable way to circumvent the tremendous hunger for high quality training data and effectively enables few-shot learning of dense segmentations.

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
