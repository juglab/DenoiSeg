from os.path import join
import os
from shutil import copy as cp
from distutils.dir_util import copy_tree as copytree
import glob
import sys
from PyInquirer import prompt, Validator, ValidationError

import json

singularity_path = "/projects/Noise2Seg/singularity/"

base_dir = "/projects/Noise2Seg"
gitrepo_path = join(base_dir, 'Noise2Seg')
base_path_data = join(base_dir, "data")
base_path_exp = join(base_dir, "experiments")

def data_path(config):
    l = glob.glob(join(base_path_data, '*'), recursive=True)
    if len(l) == 0:
        raise Exception("No training data available in {}".format(base_path_data))

    return l


class ValExpName(Validator):
    def validate(self, document):
        names = glob.glob(join(base_path_exp, '*'))
        names = [n.split('/')[-1] for n in names]

        if document.text in names:
            raise ValidationError(
                message='An experiment with this name already exists. Please choose another name.',
                cursor_position=len(document.text)
            )


class TrainFracValidator(Validator):
    def validate(self, document):
        values = document.text.split(',')
        for v in values:
            try:
                float_v = float(v)
                if float_v < 0 or float_v > 100:
                    raise ValidationError(
                        message='Enter a comma separated list of floats between 0 and 100.',
                        cursor_position=len(document.text)
                    )
            except ValueError:
                raise ValidationError(
                    message='Enter a list of floats between 0 and 100.',
                    cursor_position=len(document.text)
                )


def main():
    questions = [
        {
            'type': 'input',
            'name': 'exp_name',
            'message': 'Experiment name:',
            'validate': ValExpName
        },
        {
            'type': 'list',
            'name': 'data_path',
            'message': 'Data path:',
            'choices': data_path
        },
        {
            'type': 'input',
            'name': 'repetitions',
            'message': 'Number of repetitions',
            'default': '5',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'train_frac',
            'message': 'Training data fractions in x%:',
            'default': '0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0,64.0,100.0',
            'validate': TrainFracValidator,
            'filter': lambda val: [float(x) for x in val.split(',')]
        },
        {
            'type': 'input',
            'name': 'unet_n_depth',
            'message': 'unet_n_depth',
            'default': '4',
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'train_epochs',
            'message': 'train_epochs',
            'default': '200',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'train_batch_size',
            'message': 'train_batch_size',
            'default': '128',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'n2s_weight_seg',
            'message': 'Noise2Seg weighting for segmentation',
            'default': '1.0',
            'validate': lambda val: float(val) >= 0,
            'filter': lambda val: float(val)
        },
        {
            'type': 'input',
            'name': 'n2s_weight_denoise',
            'message': 'Noise2Seg weighting for denoising',
            'default': '1.0',
            'validate': lambda val: float(val) >= 0,
            'filter': lambda val: float(val)
        }
    ]

    config = prompt(questions)

    for run_idx in range(1, config['repetitions'] + 1):
        for train_fraction in config['train_frac']:
            run_name = config['exp_name'] + '_run' + str(run_idx)
            exp_conf = create_configs(config, run_name, seed=run_idx, train_fraction=train_fraction)

            exp_path = join(base_path_exp, run_name, "fraction_" + str(train_fraction))
            start_n2s_experiment(exp_conf, exp_path, config['data_path'])


def create_configs(config, run_name, seed, train_fraction):
    exp_conf = {
        "train_data_path": join("/data", "train", "train_data.npz"),
        "test_data_path": join("/data", "test", "test_data.npz"),
        "exp_dir": join("/notebooks", run_name),
        "fraction": train_fraction,
        "random_seed": seed,
        "model_name": run_name + "_model",
        "basedir": "/notebooks",
        "train_epochs": config['train_epochs'],
        "train_batch_size": config['train_batch_size'],
        "unet_n_depth": config['unet_n_depth'],
        "n2s_weight_seg": config['n2s_weight_seg'],
        "n2s_weight_denoise": config['n2s_weight_denoise']
    }

    return exp_conf


def start_n2s_experiment(exp_conf, exp_path, data_path):
    os.makedirs(exp_path, exist_ok=True)

    copytree(join(gitrepo_path, 'noise2seg'), join(exp_path, 'noise2seg'))
    cp(join(gitrepo_path, 'noise2seg.py'), exp_path)

    with open(join(exp_path, 'experiment.json'), 'w') as f:
        json.dump(exp_conf, f)

    slurm_script = create_slurm_script(exp_path, data_path)
    with open(join(exp_path, 'slurm.job'), 'w') as f:
        for l in slurm_script:
            f.write(l)

    os.system('chmod -R 775 ' + exp_path)

    os.system('sbatch {}'.format(join(exp_path, "slurm.job")))


def create_slurm_script(exp_path, data_path):
    script = [
        "#!/bin/bash\n",
        "#SBATCH -J Noise2Seg\n",
        "#SBATCH -o /projects/Noise2Seg/slurm_logs/slurm-%A.log  # output file\n",
        "#SBATCH -t 12:00:00       # max. wall clock time 5s\n",
        "#SBATCH -n 1          # number of tasks\n",
        "#SBATCH -N 1\n",
        "#SBATCH -c 1\n",
        "#SBATCH --partition=gpu\n",
        "#SBATCH --gres=gpu:1\n",
        "#SBATCH --exclude=r02n01,r01n01,r01n02,r01n03,r01n04,r02n22\n",
        "#SBATCH --mem=32000\n",
        "#SBATCH --export=ALL\n",
        "\n",
        "cd {}\n".format(singularity_path),
        "srun -J N2S -o {}/noise2seg.log singularity exec --nv -B user/:/run/user -B {}:/notebooks -B {}:/data {} python3 /notebooks/noise2seg.py --exp_conf /notebooks/experiment.json\n".format(exp_path, exp_path, data_path, singularity_path+"noise2seg.simg")
    ]

    return script


if __name__ == "__main__":
    main()
