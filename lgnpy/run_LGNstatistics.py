import tqdm
import multiprocessing
import cv2
import os
from lgnpy.CEandSC.lgn_statistics import lgn_statistics
from result_manager.result_manager import ResultManager
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRetinalGanglionCellSampling

def iterate(args):
    folder, filename, threshold_lgn, rgc, coc = args

    im = cv2.imread(os.path.join(folder, filename)) # Always loads images in BGR order
    im = im[:,:,::-1] # Reverse order to RGB

    if rgc:
        GCS = ToRetinalGanglionCellSampling()
        im = GCS(im)

    (ce, sc, beta, gamma) = lgn_statistics(im=im, coc=coc, file_name=None, force_recompute=True, cache=False, threshold_lgn=threshold_lgn, compute_extra_statistics=False, verbose_filename=False, verbose=False)

    return filename, ce, sc, beta, gamma

def run_LGNstatistics():
    # folder = '\\wsl.localhost\Ubuntu\home\niklas\projects\mouse_lgn\data\natural_scene_templates'
    # folder = '\\wsl.localhost\\Ubuntu\\home\\niklas\\projects\\data\\oads\\oads_arw\\tiff'
    folder = '/home/niklas/projects/data/oads/oads_arw/tiff/reduced'

    rgc = False
    coc = False

    result_manager = ResultManager(root='/home/niklas/projects/lgnpy/results')
    # addpath(folder)
    # addpath("CEandSCmatlab")

    file_types = ['tiff'] # 'jpg', 'jpeg', 'bmp', 'png', 'tif', 
    file_names = [x for x in os.listdir(folder) if x.split('.')[-1] in file_types]

    ce_results = []
    sc_results = []
    beta_results = []
    gamma_results = []
    filenames = []

    threshold_lgn = loadmat('/home/niklas/projects/lgnpy/ThresholdLGN.mat')
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

    with multiprocessing.Pool(multiprocessing.cpu_count()-4) as pool:
        results = list(tqdm.tqdm(pool.imap(iterate, [(folder, filename, threshold_lgn, rgc, coc) for filename in file_names]), total=len(file_names)))

    for (filename, ce, sc, beta, gamma) in results:
        ce_results.append(ce)
        sc_results.append(sc)
        beta_results.append(beta)
        gamma_results.append(gamma)
        filenames.append(filename)
    

    results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, 'filenames': filenames}
    result_manager.save_result(result=results, filename=f'oads{"_gcs" if rgc else ""}{"rgb" if not coc else ""}_lgn_statistics.pkl')

    # return (ce, sc, beta, gamma, filenames)

if __name__ == '__main__':
    run_LGNstatistics()