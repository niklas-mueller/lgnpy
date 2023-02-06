import cv2
import os
from lgnpy.CEandSC.lgn_statistics import lgn_statistics
from result_manager.result_manager import ResultManager
from oads_access.utils import loadmat

def run_LGNstatistics():
    # folder = '\\wsl.localhost\Ubuntu\home\niklas\projects\mouse_lgn\data\natural_scene_templates'
    # folder = '\\wsl.localhost\\Ubuntu\\home\\niklas\\projects\\data\\oads\\oads_arw\\tiff'
    folder = '/home/niklas/projects/data/oads/oads_arw/tiff'

    result_manager = ResultManager(root='/home/niklas/projects/lgnpy/results')
    # addpath(folder)
    # addpath("CEandSCmatlab")

    file_types = ['jpg', 'jpeg', 'bmp', 'png', 'tif', 'tiff']
    file_names = os.listdir(folder)

    ce_results = []
    sc_results = []
    beta_results = []
    gamma_results = []
    filenames = []

    threshold_lgn = loadmat('/home/niklas/projects/LGNstatistics-master/CEandSCmatlab/ThresholdLGN.mat')
    threshold_lgn = threshold_lgn['ThresholdLGN']

    for file_index, file_name in enumerate(file_names):
        print(file_name)
        if file_name.split('.')[-1] in file_types:
            im = cv2.imread(os.path.join(folder, file_name)) # Always loads images in BGR order
            im = im[:,:,::-1] # Reverse order to RGB


            # Instead of saving all crops to files and then computing stuff on it, create crops on the fly and 
            # save all the meta data (crop box, file name, crop index, etc.) in a yml file and only save the computed statistics
            (ce, sc, beta, gamma) = lgn_statistics(im, threshold_lgn)
            ce_results.append(ce)
            sc_results.append(sc)
            beta_results.append(beta)
            gamma_results.append(gamma)
            filenames.append(file_name)
        else:
            continue

    results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, 'filenames': filenames}
    result_manager.save_result(result=results, filename='lgn_statistics.csv')

    return (ce, sc, beta, gamma, filenames)

if __name__ == '__main__':
    run_LGNstatistics()