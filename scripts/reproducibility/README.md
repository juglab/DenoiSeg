# Reproducibility
In this sub-directory you can find all scripts and jupyter notebook 
which we used to produce the results reported in the manuscript. If
you want to run the exact code we used, you have to use release 
[v0.1.7](https://github.com/juglab/DenoiSeg/releases/tag/v0.1.7).
You will see the name `noise2seg`, the original name of the project, 
in these files. We only renamed the project to `DenoiSeg` once we 
started writing it up :)

__Note:__ These files are here for completeness and not expected to 
run out of the box on any given setup. 

## ./cluster_execution
The scripts in this sub-directory were used to run all our experiments
on the MPI-CBG cluster. All experiments were run via singularity. The 
singularity image was built with `./cluster_exectuion/singularity/noise2seg.Singularity`. 

### Build Singularity Container
1. Install Singularity
2. `sudo singularity build denoiseg.sim singularity/noise2seg.Singularity`

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

## ./dataprep
This sub-directory contains all the jupyter notebooks which we used to
prepare the training datasets. 

## ./figures
This sub-directory contains all the jupyter notebooks which we used to 
create the figures. These notebooks use TeX to render the fonts. If you
want to run them, make sure that you have TeX installed. 

## ./tables
This sub-directory contains all the jupyter notebooks which we used to
create the tables.