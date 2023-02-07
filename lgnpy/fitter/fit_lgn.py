from lgnpy.CEandSC.lgn_statistics import lgn_statistics, interp1d, regress
# conv2padded, filter_lgn, create_hist, local_cov
from result_manager.result_manager import ResultManager
from lgnpy.run_LGNstatistics import loadmat
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import zscore
import os
from sklearn.linear_model import LinearRegression
import multiprocessing
import warnings
import tqdm
from scipy.optimize import minimize

################
# Define Subfunctions

# 
def iteration(args):
    image_name, image_index, config, threshold_lgn = args
    im = Image.open(os.path.join('/home/niklas/projects/eeg_jneurosc13/allstim/', image_name))
    im = np.array(im)

    (ce, sc, _, _) = lgn_statistics(im=im, file_name=image_name, config=config, threshold_lgn=threshold_lgn)

    return ce, sc, image_index

################

def get_sc_ce(image_names, config, threshold_lgn):
    ces = np.zeros((len(image_names), 3, 1, 2))
    scs = np.zeros((len(image_names), 3, 1, 2))


    with multiprocessing.Pool() as pool:
        results = list(tqdm.tqdm(pool.imap(iteration, [(image_name, index, config, threshold_lgn) for index, image_name in enumerate(image_names)]), total=len(image_names)))
    
    for ce, sc, index in results:
        # Make sure that this is in the same order as the ERP data!
        filename = image_names[index]
        index = int(filename.split('.')[0].split('_')[1]) - 1
        ces[index, :,:,:] = ce
        scs[index, :,:,:] = sc

    return ces, scs

################
# Regression
def regression(ces, scs):
    average_colors = True

    # Get design matrix
    if average_colors:
        ce_avg = zscore(np.mean(ces[:,:,0,0], axis=-1))
        sc_avg = zscore(np.mean(scs[:,:,0,0], axis=-1))
    else:
        ce_avg = zscore(ces[:,:,0,0], axis=0)
        sc_avg = zscore(scs[:,:,0,0], axis=0)

    if len(ce_avg.shape) > 1:
        c = np.ones((ce_avg.shape[0], 1)) #* np.mean(ce_avg)
        zeros = np.zeros(ce_avg.shape)
        rep1 = np.hstack((c, ce_avg, zeros, sc_avg, zeros))
        rep2 = np.hstack((c, zeros, ce_avg, zeros, sc_avg))
        design_matrix = np.vstack((rep1, rep2))
    else:
        c = np.ones((ce_avg.shape[0])) #* np.mean(ce_avg)
        zeros = np.zeros(ce_avg.shape)
        rep1 = np.vstack((c, ce_avg, zeros, sc_avg, zeros))
        rep2 = np.vstack((c, zeros, ce_avg, zeros, sc_avg))
        design_matrix = np.hstack((rep1, rep2)).T

    r2_sensors = {}
    beta_sensors = {}
    
    # Iterate over sensors
    for sensor_index, sensor in [(i, _x) for i, _x in enumerate(sensors)]: #  if 'O' in _x
        r2_subjects = {}
        beta_subjects = {}

        # Iterate over subjects
        for subject in range(14):
            r2s = []
            betas = []

            # Iterate over time points
            for time_index in range(154):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    y = data[subject,time_index,sensor_index,:]
                r2, beta = regress(y, design_matrix=design_matrix)
                r2s.append(r2)
                betas.append(beta)

            r2_subjects[subject] = r2s
            beta_subjects[subject] = betas

    r2_sensors[sensor] = r2_subjects
    beta_sensors[sensor] = beta_subjects
    
    return (r2_sensors, beta_sensors)

################
# Full computation

def comb(image_names, threshold_lgn, config):
    ces, scs = get_sc_ce(image_names, threshold_lgn=threshold_lgn, config=config)
    return regression(ces, scs)

################

if __name__ == '__main__':

    force_recompute = True
    root = '/home/niklas/projects/eeg_jneurosc13/allstim/'

    ################
    result_manager = ResultManager(root='/home/niklas/projects/lgnpy/lgnpy/fitter/results')
    erp = loadmat('/home/niklas/projects/eeg_jneurosc13/ERPs_singlesubjects14_singletrials_behexcluded.mat')
    sensors = erp['sensors']
    data = erp['Y']
    del erp

    configs = [
            {
                'viewing_dist': 0.5
            },
            {
                'viewing_dist': 1
            },
            {
                'viewing_dist': 2
            },
            # {
            #     'viewing_dist': 5
            # }
        ]


    if force_recompute:
        ################
        # Import Data

        threshold_lgn = loadmat('/home/niklas/projects/lgnpy/ThresholdLGN.mat')
        threshold_lgn = threshold_lgn['ThresholdLGN']

        ################
        # Get Stimuli
        image_names = [f for f in os.listdir(root) if os.path.join(root, f).endswith('jpg')]

        res = []

        # for viewing_dist in [0.5, 0.8, 1.0, 1.5]:
        for config in configs:
            res.append(comb(image_names=image_names, threshold_lgn=threshold_lgn, config=config))

        ################
        result_manager.save_result(result=res, filename='full_fitting_results.pkl')

    else:
        res = result_manager.load_result(filename='full_fitting_results.pkl')

    ################
    # Visuals

    figs = []
    for sensor_index, sensor in enumerate(res[0][0].keys()):
        fig, ax = plt.subplots(len(configs), 1, figsize=(15,len(configs)*5))

        for index, _ in enumerate(configs):
            for sub in range(14):
                if -1 not in res[index][0][sensor][sub]:
                    ax[index].plot(res[index][0][sensor][sub])

            ax[index].plot(np.mean(np.array([x for _, x in res[index][0][sensor].items()]), axis=0), 'k', linewidth=4.0)

        figs.append(fig)

    result_manager.save_pdf(figs=figs, filename='r2_fitting.pdf')
    ################
    ################
    ################
    ################
    ################
    ################
    ################
    ################
    ################
    ################
    ################
    ################

    print('Done')