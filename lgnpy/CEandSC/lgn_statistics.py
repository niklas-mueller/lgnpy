import warnings
import cv2
import numpy as np
from scipy.interpolate import interp1d
from copy import deepcopy
import yaml
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages
from result_manager.result_manager import ResultManager

class LGN():

    def __init__(self, config=None, default_config_path:str='./default_config.yaml'):
        self.default_config_path = default_config_path

        self.default_config : dict

        if config is not None and type(config) == str:
            self.config_path = config
            self.config : dict
            with open(self.config_path, 'r') as f:
                self.config = yaml.load(f, Loader=yaml.UnsafeLoader)
        elif config is not None and type(config) == dict:
            self.config_path = ''
            self.config = config
        else:
            self.config_path = ''
            self.config = {}

        with open(self.default_config_path, 'r') as f:
            self.default_config = yaml.load(f, Loader=yaml.UnsafeLoader)

    def get_attr(self, name):
        if name in self.config.keys():
            return self.config[name]
        elif name in self.default_config.keys():
            return self.default_config[name]
        else:
            raise KeyError(name)

    def weibullNewtonHist(self, g, x, h):
        x_g = x ** g
        sum_x_g = np.sum(x_g*h)
        x_i = x_g / sum_x_g

        ln_x_i = np.log(x_i)

        _lambda = x_g * (np.log(x) * sum_x_g - np.sum(h *
                        x_g * np.log(x))) / (sum_x_g ** 2)
        f = 1 + np.sum(ln_x_i * h) - np.sum(x_i * ln_x_i * h)
        f_prime = np.sum(_lambda * h * (sum_x_g / x_g - ln_x_i - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return f / f_prime


    def weibullMleHist(self, ax, h):
        eps = 0.01  # precision of Newton-Raphson method's solution
        shape = 0.1  # initial value of gamma parameter

        h = h / np.sum(h)
        shape_next = shape - self.weibullNewtonHist(shape, ax, h)
        n_iteration = 0

        while np.abs(shape_next - shape) > eps:
            if np.isnan(shape_next) or np.isinf(shape_next) or shape_next > 20 or n_iteration > 30:
                break

            if shape_next <= 0:
                shape_next = 0.000001
                break

            shape = shape_next
            shape_next = shape - self.weibullNewtonHist(shape, ax, h)

            n_iteration += 1

        shape = shape_next
        scale = np.power(np.sum((np.power(ax, shape) * h)), 1/shape)

        return scale, shape


    def matlab_style_gauss2D(self, shape=(3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])

        from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python

        """
        m, n = [(ss-1.)/2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps*h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


    def local_cov(self, e, sigma):
        # break_off_sigma = 3.0
        break_off_sigma = self.get_attr('break_off_sigma')
        filter_size = np.round(break_off_sigma * sigma)
        h = self.matlab_style_gauss2D((filter_size, filter_size), sigma)

        term1 = cv2.filter2D(e**2, -1, h, borderType=cv2.BORDER_REPLICATE)
        term2 = cv2.filter2D(e, -1, h, borderType=cv2.BORDER_REPLICATE)**2
        local_std = np.sqrt(np.maximum(term1 - term2, np.zeros(term1.shape)))

        local_mean = cv2.filter2D(
            e, -1, h, borderType=cv2.BORDER_REPLICATE) + np.finfo(float).tiny

        return local_std / local_mean


    def create_hist(self, data, h_bins):

        i_bin = np.array(range(1, h_bins+1))
        hist, bins = np.histogram(data, bins=h_bins)
        delta = (np.max(data) - np.min(data)) / h_bins
        ax = ((np.min(data) + i_bin * delta) +
            (np.min(data) + i_bin-1 * delta)) / 2

        # ind = np.argwhere(h)
        ind = hist > 0
        h = hist[ind]
        ax = ax[ind]

        return ax, h


    def rgb2e(self, im):
        # convert RGB to e, el, ell

        R = im[:, :, 0]
        G = im[:, :, 1]
        B = im[:, :, 2]

        # JMG: slightly different values than in PAMI 2001 paper;
        # simply assuming correctly white balanced camera
        color_weighting_e = self.get_attr('color_weighting_e') #= [0.3, 0.58, 0.11]
        color_weighting_el = self.get_attr('color_weighting_el') #= [0.25, 0.25, -0.5]
        color_weighting_ell = self.get_attr('color_weighting_ell') #= [0.5, -0.5]

        E = (color_weighting_e[0]*R + color_weighting_e[1]*G + color_weighting_e[2]*B) / 255.0
        El = (color_weighting_el[0]*R + color_weighting_el[1]*G + color_weighting_el[2]*B) / 255.0
        Ell = (color_weighting_ell[0]*R + color_weighting_ell[1]*G + color_weighting_ell[2]*B) / 255.0

        # # As in original in PAMI 2001 paper
        # E   = (0.06*R + 0.63*G + 0.27*B ) / 255.0
        # El  = (0.3*R  + 0.04*G - 0.35*B ) / 255.0
        # Ell = (0.34*R - 0.6*G+0.17*B) / 255.0

        return E, El, Ell


    def conv2padded(self, varargin):
        # CONV2PADDED  Two-dimensional convolution with padding.
        #    Y = CONV2PADDED(X,H) applies 2D filter H to X with constant extension
        #    padding.
        #
        #    Y = CONV2PADDED(H1,H2,X) first applies 1D filter H1 along the rows and
        #    then applies 1D filter H2 along the columns.
        #
        #    If X is a 3D array, filtering is done separately on each channel.

        #  Pascal Getreuer 2009

        if len(varargin) == 2:
            x, h = varargin
            if len(h.shape) > 1:
                vertical = h.shape[0]
                horizontal = h.shape[1]
            else:
                vertical = 1
                horizontal = h.shape[0]
                h = np.resize(h, (1, h.shape[0]))

            top = int(np.ceil(vertical / 2) - 1)
            bottom = int(np.floor(vertical / 2))
            left = int(np.ceil(horizontal/2) - 1)
            right = int(np.floor(horizontal / 2))

        elif len(varargin) == 3:
            h1, h2, x = varargin
            top = int(np.ceil(len(h1)/2) - 1)
            bottom = int(np.floor(len(h1)/2))
            left = int(np.ceil(len(h2)/2)-1)
            right = int(np.floor(len(h2)/2))
        else:
            raise AttributeError()

        # pad the input image
        x_padded = np.pad(x, pad_width=[(top, bottom), (left, right)], mode='edge')

        def conv2(self, v1, v2, m, mode='same'):
            """
            Two-dimensional convolution of matrix m by vectors v1 and v2

            First convolves each column of 'm' with the vector 'v1'
            and then it convolves each row of the result with the vector 'v2'.

            from https://stackoverflow.com/questions/24231285/is-there-a-python-equivalent-to-matlabs-conv2h1-h2-a-same

            """
            tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
            return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)

        if x.ndim == 2:
            x.resize((x.shape[0], x.shape[1], 1))
            x_padded.resize((x_padded.shape[0], x_padded.shape[1], 1))

        for p in range(x.shape[2]):
            if len(varargin) == 2:
                # ans1 = convolve2d(x_padded[:,:,p], h, mode='valid')
                ans2 = cv2.filter2D(x[:,:,p], -1, h, borderType=cv2.BORDER_REPLICATE)
                x[:, :, p] = ans2
            else:
                x[:, :, p] = self.conv2(h1, h2, x_padded[:, :, p], mode='valid')

        return x


    def filter_lgn(self, im, sigma):
        # break_off_sigma = 3
        break_off_sigma = self.get_attr('break_off_sigma')
        filter_size = break_off_sigma * sigma
        x = np.array([i for i in range(-filter_size, filter_size+1)])

        gauss = 1 / (np.sqrt(2*np.pi) * sigma) * \
            np.exp((x**2) / (-2 * sigma * sigma))
        Gx = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
        Gx = Gx - sum(Gx) / len(x)
        Gx = Gx / sum(0.5 * x * x * Gx)

        Gy = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
        Gy = Gy - sum(Gy) / len(x)
        Gy = Gy / sum(0.5 * x * x * Gy)

        if im.shape[2] == 1:
            im = (im / np.max(im)).squeeze()
            Ex = self.conv2padded((deepcopy(im), Gx))
            Ey = self.conv2padded((deepcopy(im), np.matrix(Gy).H))
            e = np.sqrt(Ex**2 + Ey**2).squeeze()
            el = []
            ell = []

        else:
            e, el, ell = self.rgb2e(im)

            # im = e
            Ex = self.conv2padded((deepcopy(e), Gx))
            Ey = self.conv2padded((deepcopy(e), np.matrix(Gy).H))
            e = np.sqrt(Ex**2 + Ey**2).squeeze()

            # im = el
            Elx = self.conv2padded((deepcopy(el), Gx))
            Ely = self.conv2padded((deepcopy(el), np.matrix(Gy).H))
            el = np.sqrt(Elx**2 + Ely**2).squeeze()

            # im = ell
            Ellx = self.conv2padded((deepcopy(ell), Gx))
            Elly = self.conv2padded((deepcopy(ell), np.matrix(Gy).H))
            ell = np.sqrt(Ellx**2 + Elly**2).squeeze()

        return e, el, ell

def regress(y, design_matrix):
    mask = np.isnan(y)
    x = design_matrix[~mask]
    y = zscore(y[~mask])
    if x.shape[0] == 0 or y.shape[0] == 0:
        # r2s.append(-1)
        # continue
        return -1, 0

    lin_reg = LinearRegression().fit(x, y)
    r2 = lin_reg.score(x,y)
    beta = lin_reg.coef_
    return r2, beta


def lgn_statistics(im, file_name:str, threshold_lgn, config=None, verbose: bool = False, compute_extra_statistics: bool = False, crop_masks: list = [], force_recompute:bool=False, cache:bool=True):

    result_manager = ResultManager(root='/home/niklas/projects/lgnpy/cache', verbose=False)

    lgn = LGN(config=config, default_config_path='/home/niklas/projects/lgnpy/lgnpy/CEandSC/default_config.yml')

    # if verbose:
    print(f"Computing LGN statistics for {file_name}")
    # Check if file exists
    file_name = f"results_{file_name}.npz"
    if file_name is not None and not force_recompute:
        try:
            results = result_manager.load_result(filename=file_name)
        except:
            results = None
    else:
        results = None
    
    if type(im) is str:
        im = cv2.imread(im)

    #
    # Set image parameters
    #

    if im.shape[-1] == 2:
        IMTYPE = 1  # Gray
    elif im.shape[-1] == 3:
        IMTYPE = 2  # Color
    else:
        IMTYPE = 1
        # im = im.reshape((im.shape) + (1,))
        # print(im.shape)        

    imsize = im.shape[:2]

    #######################################################
    # Set parameters for field of view
    #######################################################
    def get_field_of_view(lgn, imsize, viewing_dist):
        # if viewing_dist is None:
        #     viewing_dist = lgn.get_attr('viewing_dist')
        dot_pitch = lgn.get_attr('dot_pitch')
        fov_beta = lgn.get_attr('fov_beta')
        fov_gamma = lgn.get_attr('fov_gamma')

        fovx = round(imsize[1]/2)          # x-pixel loc. of fovea center
        fovy = round(imsize[0]/2)          # y-pixel loc. of fovea center
        # ex and ey are the x- and y- offsets of each pixel compared to
        # the point of focus (fovx,fovy) in pixels.
        ex, ey = np.meshgrid(np.arange(start=-fovx+1, stop=imsize[1]-fovx+1),
                            np.arange(start=-fovy+1, stop=imsize[0]-fovy+1))
        # eradius is the radial distance between each point and the point
        # of gaze.  This is in meters.
        eradius = dot_pitch * np.sqrt(ex**2+ey**2)
        del ex, ey
        # calculate ec, the eccentricity from the foveal center, for each
        # point in the image.  ec is in degrees.
        ec = 180*np.arctan(eradius / viewing_dist)/np.pi
        # select the pixels that fall within the input visual field of view
        imfovbeta = (ec < fov_beta)
        imfovgamma = (ec < fov_gamma)

        return imfovbeta, imfovgamma

    viewing_dist = lgn.get_attr('viewing_dist')
    imfovbeta, imfovgamma = get_field_of_view(lgn=lgn, imsize=imsize, viewing_dist=viewing_dist)

    # We need adjusted imfovbeta, imfovgamma for the crops
    imfovbeta_crops = [[] for _ in range(len(crop_masks))]
    imfovgamma_crops = [[] for _ in range(len(crop_masks))]
    for index, mask in enumerate(crop_masks):
        _x = mask.sum(axis=0)
        mask_height = _x[_x > 0][0]
        _x = mask.sum(axis=1)
        mask_width = _x[_x > 0][0]
        mask_imsize = (mask_height, mask_width)

        mask_viewing_dist = np.mean(np.array(mask_imsize) / np.array(imsize)) * viewing_dist
        _imbeta, _imgamma = get_field_of_view(lgn=lgn, imsize=mask_imsize, viewing_dist=mask_viewing_dist)
        imfovbeta_crops[index] = _imbeta
        imfovgamma_crops[index] = _imgamma


    # (color_channels, (full+boxes), center-peripherie)
    ce = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    sc = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    beta = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    gamma = np.zeros((im.shape[-1], 1+len(crop_masks), 2))

    par1, par2, par3, mag1, mag2, mag3 = get_edge_maps(im, file_name, threshold_lgn, verbose, force_recompute, cache, result_manager, lgn, results, IMTYPE, imsize)

    ##############
    # Compute Feature Energy and Spatial Coherence
    ##############

    def get_crop_masks(fov, mask):
        _start_x = mask.sum(axis=1).argmax()
        _start_y = mask.sum(axis=0).argmax()
        c_mask = np.zeros(mask.shape, dtype=np.bool8)
        c_mask[_start_x:_start_x+fov.shape[0], _start_y:_start_y+fov.shape[1]] = fov * mask[mask].reshape(fov.shape)
        c_mask_peri = np.zeros(mask.shape, dtype=np.bool8)
        c_mask_peri[_start_x:_start_x+fov.shape[0], _start_y:_start_y+fov.shape[1]] = (~fov) * mask[mask].reshape(fov.shape)
        return c_mask, c_mask_peri

    if verbose:
        print("Compute CE")

    magnitude = np.abs(par1[imfovbeta])
    # Full scene, red/gray
    ce[0, 0, 0] = np.mean(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ce[1, 0, 0] = np.mean(magnitude)
        magnitude = np.abs(par3[imfovbeta])
        ce[2,0,0] = np.mean(magnitude)

    if compute_extra_statistics:
        # Peripherie
        peri = np.mean(np.abs(par1[~imfovbeta]))
        ce[0, 0, 1] = peri

        if IMTYPE == 2:
            peri = np.mean(np.abs(par2[~imfovbeta]))
            ce[1, 0, 1] = peri
            peri = np.mean(np.abs(par3[~imfovbeta]))
            ce[2, 0, 1] = peri
        
        # Custom boxes (crops)
        for mask_index, mask in enumerate(crop_masks):
            c_mask, c_mask_peri = get_crop_masks(imfovbeta_crops[mask_index], mask)

            box_center = np.mean(np.abs(par1[c_mask]))
            ce[0, mask_index+1, 0] = box_center
            box_peri = np.mean(np.abs(par1[c_mask_peri]))
            ce[0, mask_index+1, 1] = box_peri

            if IMTYPE == 2:
                box_center = np.mean(np.abs(par2[c_mask]))
                ce[1, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par2[c_mask_peri]))
                ce[1, mask_index+1, 1] = box_peri
                box_center = np.mean(np.abs(par3[c_mask]))
                ce[2, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par3[c_mask_peri]))
                ce[2, mask_index+1, 1] = box_peri


    if verbose:
        print("Compute SC")
    magnitude = np.abs(mag1[imfovgamma])
    sc[0,0,0] = np.mean(magnitude) / np.std(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        sc[1,0,0] = np.mean(magnitude) / np.std(magnitude)
        magnitude = np.abs(mag3[imfovgamma])
        sc[2,0,0] = np.mean(magnitude) / np.std(magnitude)

    if compute_extra_statistics:
        # Peripherie
        peri = np.abs(mag1[~imfovgamma])
        sc[0, 0, 1] = np.mean(peri) / np.std(peri)

        if IMTYPE == 2:
            peri = np.abs(mag2[~imfovgamma])
            sc[1, 0, 1] = np.mean(peri) / np.std(peri)
            peri = np.abs(mag3[~imfovgamma])
            sc[2, 0, 1] = np.mean(peri) / np.std(peri)
        
        # Custom boxes (crops)
        for mask_index, mask in enumerate(crop_masks):
            c_mask, c_mask_peri = get_crop_masks(imfovgamma_crops[mask_index], mask)

            box_center = np.abs(mag1[c_mask])
            sc[0, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
            box_peri = np.abs(mag1[c_mask_peri])
            sc[0, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

            if IMTYPE == 2:
                box_center = np.abs(mag2[c_mask])
                sc[1, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag2[c_mask_peri])
                sc[1, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)
                box_center = np.abs(mag3[c_mask])
                sc[2, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag3[c_mask_peri])
                sc[2, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

    #################
    # Compute Weibull parameters beta and gamma
    #################

    if verbose:
        print("Compute Weibull parameters beta")

    # n_bins = 1000
    n_bins = lgn.get_attr('n_bins_weibull')
    magnitude = np.abs(par1[imfovbeta])
    ax, h = lgn.create_hist(magnitude, n_bins)
    # beta.append(lgn.weibullMleHist(ax, h)[0])
    beta[0,0,0] = lgn.weibullMleHist(ax, h)[0]

    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ax, h = lgn.create_hist(magnitude, n_bins)
        # beta.append(lgn.weibullMleHist(ax, h)[0])
        beta[1,0,0] = lgn.weibullMleHist(ax, h)[0]

        magnitude = np.abs(par3[imfovbeta])
        ax, h = lgn.create_hist(magnitude, n_bins)
        # beta.append(lgn.weibullMleHist(ax, h)[0])
        beta[2,0,0] = lgn.weibullMleHist(ax, h)[0]

    # Custom boxes (crops)
    for mask_index, mask in enumerate(crop_masks):
        c_mask, c_mask_peri = get_crop_masks(imfovbeta_crops[mask_index], mask)

        box_center = np.abs(par1[c_mask])
        ax, h = lgn.create_hist(box_center, n_bins)
        beta[0, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
        box_peri = np.abs(par1[c_mask_peri])
        ax, h = lgn.create_hist(box_peri, n_bins)
        beta[0, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

        if IMTYPE == 2:
            box_center = np.abs(par2[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[1, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
            box_peri = np.abs(par2[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[1, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

            box_center = np.abs(par3[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[2, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[0]
            box_peri = np.abs(par3[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[2, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[0]

    if verbose:
        print("Compute Weibull parameters gamma")
    magnitude = np.abs(mag1[imfovgamma])
    ax, h = lgn.create_hist(magnitude, n_bins)
    gamma[0,0,0] = lgn.weibullMleHist(ax, h)[1]

    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        ax, h = lgn.create_hist(magnitude, n_bins)
        gamma[1,0,0] = lgn.weibullMleHist(ax, h)[1]

        magnitude = np.abs(mag3[imfovgamma])
        ax, h = lgn.create_hist(magnitude, n_bins)
        gamma[2,0,0] = lgn.weibullMleHist(ax, h)[1]

    # Custom boxes (crops)
    for mask_index, mask in enumerate(crop_masks):
        c_mask, c_mask_peri = get_crop_masks(imfovgamma_crops[mask_index], mask)

        box_center = np.abs(mag1[c_mask])
        ax, h = lgn.create_hist(box_center, n_bins)
        beta[0, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
        box_peri = np.abs(mag1[c_mask_peri])
        ax, h = lgn.create_hist(box_peri, n_bins)
        beta[0, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

        if IMTYPE == 2:
            box_center = np.abs(mag2[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[1, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
            box_peri = np.abs(mag2[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[1, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

            box_center = np.abs(mag3[c_mask])
            ax, h = lgn.create_hist(box_center, n_bins)
            beta[2, mask_index+1, 0] = lgn.weibullMleHist(ax, h)[1]
            box_peri = np.abs(mag3[c_mask_peri])
            ax, h = lgn.create_hist(box_peri, n_bins)
            beta[2, mask_index+1, 1] = lgn.weibullMleHist(ax, h)[1]

    return (ce, sc, beta, gamma)

def get_edge_maps(im, file_name, threshold_lgn, lgn, IMTYPE, imsize, verbose:bool=False, force_recompute:bool=False, results=None, cache:bool=False, result_manager:ResultManager=None):
    if force_recompute or results is None:
        ######
        # Computing edges
        ######
        par1 = np.zeros(imsize)
        par2 = np.zeros(imsize)
        par3 = np.zeros(imsize)
        mag1 = np.zeros(imsize)
        mag2 = np.zeros(imsize)
        mag3 = np.zeros(imsize)

        # par_sigmas = [48, 24, 12, 6, 3]
        # mag_sigmas = [64, 32, 16, 8, 4]
        parvo_sigmas = lgn.get_attr('parvo_sigmas')
        magno_sigmas = lgn.get_attr('magno_sigmas')

        interpolation_sigmas = lgn.get_attr('interpolation_sigmas')
        eps = lgn.get_attr('eps')

        for iteration_index, sigma_iterations in enumerate(np.array([parvo_sigmas, magno_sigmas])):
            for _, sigma in enumerate(sigma_iterations):
                if verbose:
                    print(f"Sigma: {sigma}")

                if verbose:
                    print('Interpolate')
                sigmas = np.array(interpolation_sigmas)
                v1 = np.squeeze(threshold_lgn[:, 0])
                t1_interp = interp1d(sigmas, v1, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
                t1 = t1_interp(sigma)
                v2 = np.squeeze(threshold_lgn[:, 1])
                t2_interp = interp1d(sigmas, v2, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
                t2 = t2_interp(sigma)
                v3 = np.squeeze(threshold_lgn[:, 2])
                t3_interp = interp1d(sigmas, v3, kind='linear',
                                    bounds_error=False, fill_value=np.nan)
                t3 = t3_interp(sigma)

                if verbose:
                    print("Filter LGN")
                o1, o2, o3 = lgn.filter_lgn(im, sigma)

                if verbose:
                    print("Local COV 1")
                s1 = lgn.local_cov(o1, sigma)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    e1 = ((o1 * np.max(o1)) / (o1 + np.max(o1) * s1))
                minm1 = e1 - t1
                index1 = (minm1 > eps)
                if iteration_index == 0:
                    par1[index1] = minm1[index1]
                elif iteration_index == 1:
                    mag1[index1] = minm1[index1]

                if IMTYPE == 2:
                    if verbose:
                        print("Local COV 2")
                    s2 = lgn.local_cov(o2, sigma)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        e2 = ((o2 * np.max(o2)) / (o2 + np.max(o2) * s2))
                    minm2 = e2 - t2
                    index2 = (minm2 > eps)
                    if iteration_index == 0:
                        par2[index2] = minm2[index2]
                    elif iteration_index == 1:
                        mag2[index2] = minm2[index2]

                    if verbose:
                        print("Local COV 3")
                    s3 = lgn.local_cov(o3, sigma)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        e3 = ((o3 * np.max(o3)) / (o3 + np.max(o3) * s3))
                    minm3 = e3 - t3
                    index3 = (minm3 > eps)
                    if iteration_index == 0:
                        par3[index3] = minm3[index3]
                    elif iteration_index == 1:
                        mag3[index3] = minm3[index3]

    
        if cache:
            # results = np.array((par1, par2, par3, mag1, mag2, mag3))
            # result_manager.save_result(result=results, filename=file_name, overwrite=True)
            results = {"par1": par1, "par2": par2, "par3": par3, "mag1": mag1, "mag2": mag2, "mag3": mag3}
            result_manager.save_result(result=results, filename=file_name, overwrite=True)
            del results
    else:
        # par1, par2, par3, mag1, mag2, mag3 = results
        par1, par2, par3, mag1, mag2, mag3 = results['par1'], results['par2'], results['par3'], results['mag1'], results['mag2'], results['mag3']
    return par1,par2,par3,mag1,mag2,mag3
