import numpy as np


def get_measure(pre, exp, run=1, fraction=0.5, measure='SEG', score_type='validation_', 
                path_str='/home/prakash/Desktop/fileserver_Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'):
    """
    Load scores of one experiment.
    
    Parameters:
    pre : str
        Prefix of the experiment.
    exp : str
        Experiment name.
    fraction: float
        Used fraction of annotated training data.
    measure : str
        Used evaluation measure (SEG or AP). Default: 'SEG'
    score_type : str
        Score on validation ('validation_') or test ('') data. Default: 'validation_'
    path_str : str
        Path to the experiments, requires placeholders for 'pre', 'exp', 'fraction', 'score_type'. Default: '/home/tibuch/Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'
        
    Return:
        float: Requested score if available else None.
    """
    with open(path_str.format(pre, exp, run, fraction, score_type)) as f:
        line = f.readline()
        while line:
            line = line.strip().split(',')
            if line[0] == measure:
                return float(line[1])
            line = f.readline()
    return None


def read_Noise2Seg_results(pre, exp, measure='SEG', runs=[1,2,3,4,5], 
                           fractions=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0], score_type='validation_',
                           path_str='/home/prakash/Desktop/fileserver_Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'):
    """
    Load scores of a given set of experiments and compute mean and standard error of the mean over all runs. 
    
    Parameters:
    pre : str
        Prefix of the experiment.
    exp : str
        Experiment name.
    measure : str
        Used evaluation measure (SEG or AP). Default: 'SEG'
    runs : [int]
        List of runs. Default: [1,2,3,4,5]
    fractions: [float]
        List of fractions. Default: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
    score_type : str
        Score on validation ('validation_') or test ('') data. Default: 'validation_'
    path_str : str
        Path to the experiments, requires placeholders for 'pre', 'exp', 'fraction', 'score_type'. Default: '/home/tibuch/Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'
        
    Return:
        array(float): 
            1. column: fraction, 2. column: mean score, 3. column: std error of the mean
    """
    stats = []
    
    for frac in fractions:
        scores = []
        for r in runs:
            scores.append(get_measure(pre, exp, run=r, fraction=frac, measure=measure, score_type=score_type, path_str=path_str))
        
        scores = np.array(scores)
        stats.append([frac, np.mean(scores), np.std(scores)/np.sqrt(scores.shape[0])])
    
    return np.array(stats)


def read_Noise2Seg_bestAlpha_results(pre, exp, measure='SEG', runs=[1,2,3,4,5],
                                     fractions=[0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0], 
                                     alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                     path_str='/home/prakash/Desktop/fileserver_Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'):
    """
    Load the experiment which performs best on the test-data. This computes an upper-bound of our method. 
    
    Parameters:
    pre : str
        Prefix of the experiment.
    exp : str
        Experiment name.
    measure : str
        Used evaluation measure (SEG or AP). Default: 'SEG'
    runs : [int]
        List of runs. Default: [1,2,3,4,5]
    fractions: [float]
        List of fractions. Default: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 100.0]
    alphas: [float]
        List of alphas over which the search should go. Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    path_str : str
        Path to the experiments, requires placeholders for 'pre', 'exp', 'fraction', 'score_type'. Default: '/home/tibuch/Noise2Seg/experiments/{}_{}_run{}/fraction_{}/{}scores.csv'
        
    Return:
        array(float): 
            1. column: fraction, 2. column: mean score, 3. column: std error of the mean
        array(float):
            Array of each alpha which contributed to this upper bound.
    """
    stats = []
    best_alphas = []
    for frac in fractions:
        scores = []
        best_alphas_fraction = []
        for r in runs:
            best_score = 0
            best_alpha = 0
            for alpha in alphas:
                score = get_measure(pre + str(alpha), exp, run=r, fraction=frac, measure=measure, score_type="", path_str=path_str)
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            
            # read best score from test-data with this best_alpha
            best_alphas_fraction.append(best_alpha)
            scores.append(best_score)
        best_alphas.append(best_alphas_fraction)    
            
        scores = np.array(scores)
        stats.append([frac, np.mean(scores), np.std(scores)/np.sqrt(scores.shape[0])])
        
    return np.array(stats), np.array(best_alphas)


def fraction_to_abs(fracs, max_num_imgs=3800):
    """
    Convert fractions to absolute number of images.
    """
    return np.round(max_num_imgs*fracs/100)


def load_vanillaBaseline_n0(path_str='/home/prakash/Desktop/fileserver_Noise2Seg/VoidSeg_Baselines/finDepth4_dsb_n0_run{}baseline/train_{}/seg_scores.dat'):
    """
    Code to load the vanilla baseline from VoidSeg.
    
    Parameters:
    name: str
        Name of the experiment/.txt file. 
    """
    content = []
    with open('/home/tibuch/Noise2Seg/VoidSeg_Baselines/machine_readable/' + name) as f:
        line = f.readline()
        while line:
            
            content.append([float(x) for x in line.strip().split(" ")])
            line = f.readline()
    return np.array(content)

def read_voidseg_results(name):
    content = []
    with open('/home/prakash/Desktop/fileserver_Noise2Seg/VoidSeg_Baselines/machine_readable/' + name) as f:
        line = f.readline()
        while line:
            
            content.append([float(x) for x in line.strip().split(" ")])
            line = f.readline()
    return np.array(content)


def cm2inch(*tupl, scale=3):
    """
    Convert cm to inch and scale it up.
    
    Parameters:
    *tupl: *tuple(floats)
        Measures in cm to convert and scale.
    scale: float
        Scale factor. Default: 3
    """
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(scale * i/inch for i in tupl[0])
    else:
        return tuple(scale * i/inch for i in tupl)
