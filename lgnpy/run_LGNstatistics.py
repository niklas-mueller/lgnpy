import os
nproc = 5

os.environ["OMP_NUM_THREADS"] = str(nproc)
os.environ["OPENBLAS_NUM_THREADS"] = str(nproc)
os.environ["MKL_NUM_THREADS"] = str(nproc)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(nproc)

import tqdm
import multiprocessing
import cv2
from PIL import Image
from lgnpy.CEandSC.lgn_statistics import lgn_statistics
from result_manager.result_manager import ResultManager
from oads_access.utils import loadmat
from pytorch_utils.pytorch_utils import ToRetinalGanglionCellSampling, convert_from_cv2_to_image, convert_from_image_to_cv2

def iterate(args):
    folder, filename, config, threshold_lgn, rgc, coc, imsize, home_path = args

    im = cv2.imread(os.path.join(folder, filename)) # Always loads images in BGR order
    im = im[:,:,::-1] # Reverse order to RGB

    # if not high_res:
    im = cv2.resize(src=im, dsize=imsize)

    if rgc:
        GCS = ToRetinalGanglionCellSampling(out_size=max(im.shape))
        im = GCS(im)

    (ce, sc, beta, gamma) = lgn_statistics(im=im, coc=coc, config=config, home_path=home_path, file_name=None, force_recompute=True, cache=False, threshold_lgn=threshold_lgn, compute_extra_statistics=False, verbose_filename=False, verbose=False)

    return filename, ce, sc, beta, gamma

def run_LGNstatistics(rgc, coc, fov_gamma=5, viewing_dist=1, imsize=tuple):
    # folder = '\\wsl.localhost\Ubuntu\home\niklas\projects\mouse_lgn\data\natural_scene_templates'
    # folder = '\\wsl.localhost\\Ubuntu\\home\\niklas\\projects\\data\\oads\\oads_arw\\tiff'
    home_path = os.path.expanduser('~')
    oads_basedir = f'/home/Public/Datasets/oads'

    # high_res = True
    if imsize == (5496, 3672):
        high_res = True
    else:
        high_res = False

    folder = f'{oads_basedir}/oads_arw/tiff'#{"" if high_res else "/reduced"}'

    # rgc = False
    # coc = True

    config = {
        'fov_gamma': fov_gamma,
        'viewing_dist': viewing_dist,
        'dot_pitch': 0.000276,
    }

    result_path = f'{home_path}/projects/lgnpy/results/correct_viewing_dist'
    
    # if config['fov_gamma'] > 5:
    #     result_path = os.path.join(result_path, f'larger_sc-{str(config["fov_gamma"])}')

    # if config['viewing_dist'] > 1:
    #     result_path = os.path.join(result_path, f'larger_viewing_dist-{str(config["viewing_dist"])}')



    result_manager = ResultManager(root=result_path)
    # addpath(folder)
    # addpath("CEandSCmatlab")

    file_types = ['tiff'] # 'jpg', 'jpeg', 'bmp', 'png', 'tif', 
    file_names = [x for x in os.listdir(folder) if x.split('.')[-1] in file_types]

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

    with multiprocessing.Pool(8) as pool:
        results = list(tqdm.tqdm(pool.imap(iterate, [(folder, filename, config, threshold_lgn, rgc, coc, imsize, home_path) for filename in file_names]), total=len(file_names)))

    for (filename, ce, sc, beta, gamma) in results:
        ce_results.append(ce)
        sc_results.append(sc)
        beta_results.append(beta)
        gamma_results.append(gamma)
        filenames.append(filename)
    

    results = {'CE': ce_results, 'SC': sc_results, 'Beta': beta_results, 'Gamma': gamma_results, 'filenames': filenames}
    result_manager.save_result(result=results, filename=f'oads{"_gcs" if rgc else ""}{"_rgb" if not coc else ""}{"_high_res" if high_res else ""}{"x".join([str(x) for x in imsize])}_lgn_statistics.pkl')

    # return (ce, sc, beta, gamma, filenames)

if __name__ == '__main__':
    # run_LGNstatistics()
    # run_LGNstatistics(rgc=False, coc=True, fov_gamma=5, viewing_dist=2/10, imsize=(2155//10, 1440//10))
    # run_LGNstatistics(rgc=False, coc=True, fov_gamma=5, viewing_dist=2/20, imsize=(2155//20, 1440//20))
    run_LGNstatistics(rgc=True, coc=True, fov_gamma=5, viewing_dist=2, imsize=(2155, 1440))
    run_LGNstatistics(rgc=True, coc=True, fov_gamma=5, viewing_dist=3.5, imsize=(3400, 2271))
    # run_LGNstatistics(rgc=False, coc=True, fov_gamma=5, viewing_dist=5, imsize=(5496, 3672))
    # run_LGNstatistics(rgc=False, coc=True, fov_gamma=5, viewing_dist=2/4, imsize=(2155//4, 1440//4))
    