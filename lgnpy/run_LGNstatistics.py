import tqdm
import multiprocessing
import cv2
import os
from lgnpy.CEandSC.lgn_statistics import lgn_statistics
from result_manager.result_manager import ResultManager
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRetinalGanglionCellSampling
from mne import read_epochs
import pandas as pd
import numpy as np

def iterate(args):
    folder, filename, config, threshold_lgn, rgc, coc, reduce_factor = args

    im = cv2.imread(os.path.join(folder, filename)) # Always loads images in BGR order
    im = im[:,:,::-1] # Reverse order to RGB
    im = cv2.resize(im, (0,0), fx=reduce_factor, fy=reduce_factor) 

    if rgc:
        GCS = ToRetinalGanglionCellSampling(out_size=max(im.shape))
        im = GCS(im)

    (ce, sc, beta, gamma) = lgn_statistics(im=im, coc=coc, config=config, file_name=None, force_recompute=True, cache=False, threshold_lgn=threshold_lgn, compute_extra_statistics=False, verbose_filename=False, verbose=False)

    return filename, ce, sc, beta, gamma

def get_subject_filenames(home_path, sub):
    if 'niklas' in home_path:
        eeg_dir = f'/mnt/z/Projects/2023_Scholte_FMG1441/Data/sub_{sub}/Preprocessed epochs/sub_{sub}-OC&CSD-AutoReject-epo.fif'
    elif 'nmuller' in home_path:
        eeg_dir = f'{home_path}/projects/data/oads_eeg/sub_{sub}/sub_{sub}-OC&CSD-AutoReject-epo.fif'

    epochs = read_epochs(fname=eeg_dir, preload=False)

    channel_names = epochs.ch_names
    visual_channel_names = ['O1', 'O2', 'Oz', 'Iz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'PO3', 'PO7', 'POz', 'PO4', 'PO8']
    visual_channel_indices = [i for i in range(len(channel_names)) if channel_names[i] in visual_channel_names]

    if 'niklas' in home_path:
        event_ids = pd.read_csv(f'/mnt/z/Projects/2023_Scholte_FMG1441/EventsID_Dictionary.csv', header=None)
    elif 'nmuller' in home_path:
        event_ids = pd.read_csv(f'/home/nmuller/projects/data/oads_eeg/EventsID_Dictionary.csv', header=None)
    event_ids = {id: filename for _, (filename, id) in event_ids.iterrows()}

    # target_filenames = os.listdir(f'/mnt/z/Projects/2023_Scholte_FMG1441/Stimuli/reduced/Targets (reduced)')
    # target_filenames = [x for x in target_filenames]
    target_filenames = ['0434262454981c92.tiff', '0c6c2c66e61e3133.tiff', '0e2e2f2931313d37.tiff', '1332a2d4c4c14322.tiff', '202024d9b333968c.tiff', '24247432670b3131.tiff', '2c2064d163c52c78.tiff', '42431879e191d3c3.tiff', '61662ece1f6e7870.tiff', '7890918716766325.tiff', '85cc4e533959b1e1.tiff', '8ece4062eee68292.tiff', '93939b9a101e87e0.tiff', '948cacbc94b42474.tiff', 'af8f3a3939313171.tiff', 'b23232b232332361.tiff', 'b371f8ecf4e0f0f0.tiff', 'bce2d2d2c393e3b2.tiff', 'c77131717179f173.tiff', 'c93b3b2b29b9b8f1.tiff', 'cc8c9290bcbcf4fc.tiff', 'dad8acb8b82e36b1.tiff', 'e2f8f8fcf8f8f8f8.tiff', 'e6d8d40438c0c8e2.tiff', 'e6e648c62ebacc38.tiff', 'ef3632c2c1476e78.tiff', 'f4ca83dc6731310d.tiff']

    epoch_filenames = [event_ids[x] for x in epochs.events[:, 2] if x not in target_filenames]

    return epoch_filenames

def run_LGNstatistics(rgc, coc, reduce_factor=1, config={'fov_gamma':5, 'viewing_dist':1}, sub=None):
    # folder = '\\wsl.localhost\Ubuntu\home\niklas\projects\mouse_lgn\data\natural_scene_templates'
    # folder = '\\wsl.localhost\\Ubuntu\\home\\niklas\\projects\\data\\oads\\oads_arw\\tiff'
    home_path = os.path.expanduser('~')
    folder = f'{home_path}/projects/data/oads/oads_arw/tiff/reduced'

    # rgc = True
    # coc = True

    # reduce_factor = 20

    # config = {
    #     'fov_gamma': fov_gamma,
    #     'viewing_dist': viewing_dist,
    #     'dot_pitch': 0.000276,
    # }

    result_path = f'{home_path}/projects/lgnpy/results/larger_sc_ce'

    # if config['fov_gamma'] > 5:
    #     result_path = os.path.join(result_path, f'larger_sc-{str(config["fov_gamma"])}')
    #     # result_manager = ResultManager(root=f'{home_path}/projects/lgnpy/results/larger_sc-{str(config["fov_gamma"])}')

    # if config['viewing_dist'] > 1:
    #     result_path = os.path.join(result_path, f'larger_viewing_dist-{str(config["viewing_dist"])}')

    # else:
    #     result_manager = ResultManager(root=f'{home_path}/projects/lgnpy/results/')
    result_manager = ResultManager(root=result_path)


    file_types = ['tiff'] # 'jpg', 'jpeg', 'bmp', 'png', 'tif', 
    file_names = [x for x in os.listdir(folder) if x.split('.')[-1] in file_types]

    # Compute only for the images that were used in the experiment
    experiment_filenames = os.listdir('/mnt/z/Projects/2023_Scholte_FMG1441/Stimuli/reduced')
    file_names = [x for x in file_names if x in experiment_filenames]

    if sub is not None:
        epoch_filenames = []
        if type(sub) is int:
            epoch_filenames = get_subject_filenames(home_path=home_path, sub=sub)
        elif type(sub) is list or type(sub) is np.ndarray:
            for _sub in sub:
                epoch_filenames.extend(get_subject_filenames(home_path=home_path, sub=_sub))
        
        file_names = [x for x in file_names if x in epoch_filenames]

    ce_results = []
    sc_results = []
    beta_results = []
    gamma_results = []
    filenames = []

    threshold_lgn = loadmat(f'{home_path}/projects/lgnpy/ThresholdLGN.mat')
    threshold_lgn = threshold_lgn['ThresholdLGN']

    # for file_index, file_name in enumerate(file_names):
        # im = cv2.imread(os.path.join(folder, file_name)) # Always loads images in BGR order
        # im = im[:,:,::-1] # Reverse order to RGB
        # Instead of saving all crops to files and then computing stuff on it, create crops on the fly and 
        # save all the meta data (crop box, file name, crop index, etc.) in a yml file and only save the computed statistics
        # (ce, sc, beta, gamma) = lgn_statistics(im=im, file_name=None, threshold_lgn=threshold_lgn, compute_extra_statistics=False, verbose_filename=False, verbose=False)
        # ce_results.append(ce)
        # sc_results.append(sc)
        # beta_results.append(beta)
        # gamma_results.append(gamma)
        # filenames.append(file_name)

    with multiprocessing.Pool(16) as pool:
        results = list(tqdm.tqdm(pool.imap(iterate, [(folder, filename, config, threshold_lgn, rgc, coc, 1/reduce_factor) for filename in file_names]), total=len(file_names)))

    for (filename, ce, sc, beta, gamma) in results:
        ce_results.append(ce)
        sc_results.append(sc)
        beta_results.append(beta)
        gamma_results.append(gamma)
        filenames.append(filename)
    

    results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, 'filenames': filenames, 'config': config}
    result_manager.save_result(result=results, filename=f'oads{"_gcs" if rgc else ""}{"_rgb" if not coc else ""}{"_reduced-" + str(reduce_factor) if reduce_factor > 1 else ""}_lgn_statistics.pkl')

    # return (ce, sc, beta, gamma, filenames)

if __name__ == '__main__':
    run_LGNstatistics(rgc=True, coc=True, reduce_factor=4, config={'fov_gamma': 20, 'fov_beta': 3, 'viewing_dist': 2}, sub=[5,6])
    print(1)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=1, config={'fov_gamma': 5.5, 'fov_beta': 3, 'viewing_dist': 2}, sub=[5,6])
    # print(1)

    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=1, fov_gamma=5, viewing_dist=2)
    # print(1)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=1, fov_gamma=5, viewing_dist=2)
    # print(2)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=40, fov_gamma=5, viewing_dist=2/40)
    # print(3)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=4, fov_gamma=5, viewing_dist=2/4)
    # print(3)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=20, fov_gamma=5, viewing_dist=2/20)
    # print(3)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=20, fov_gamma=5, viewing_dist=2/20)
    # print(3)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=4, fov_gamma=5, viewing_dist=2/4)
    # print(4)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=10, fov_gamma=5, viewing_dist=2/10)
    # print(5)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=10, fov_gamma=5, viewing_dist=2/10)
    # print(6)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=20, fov_gamma=5, viewing_dist=2/20)
    # print(7)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=20, fov_gamma=5, viewing_dist=2/20)
    # print(8)

    ############ Done ###################
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=1, fov_gamma=10)
    # print(1)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=1, fov_gamma=15)
    # print(1)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=1, fov_gamma=20)
    # print(2)
    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=1, fov_gamma=25)
    # print(3)
    # run_LGNstatistics(rgc=True, coc=False, reduce_factor=1, fov_gamma=15)
    # print(4)
    # run_LGNstatistics(rgc=True, coc=False, reduce_factor=1, fov_gamma=20)
    # print(5)
    # run_LGNstatistics(rgc=True, coc=False, reduce_factor=1, fov_gamma=25)
    # print(6)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=1, fov_gamma=15)
    # print(7)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=1, fov_gamma=20)
    # print(8)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=1, fov_gamma=25)
    # print(9)

    # run_LGNstatistics(rgc=True, coc=True, reduce_factor=25, fov_gamma=5)
    # print(13)
    # run_LGNstatistics(rgc=False, coc=True, reduce_factor=25, fov_gamma=5)
    # print(14)
    ###################################

    
    # run_LGNstatistics(rgc=False, coc=False, reduce_factor=1, fov_gamma=15)
    # print(10)
    # run_LGNstatistics(rgc=False, coc=False, reduce_factor=1, fov_gamma=20)
    # print(11)
    # run_LGNstatistics(rgc=False, coc=False, reduce_factor=1, fov_gamma=25)
    # print(12)


    # run_LGNstatistics(rgc=True, coc=False, reduce_factor=25, fov_gamma=5)
    # print(15)
    # run_LGNstatistics(rgc=False, coc=False, reduce_factor=25, fov_gamma=5)
    # print(16)