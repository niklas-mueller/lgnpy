import warnings
import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from copy import deepcopy

from matplotlib.backends.backend_pdf import PdfPages
from result_manager.result_manager import ResultManager


def weibullNewtonHist(g, x, h):
    x_g = x ** g
    sum_x_g = np.sum(x_g*h)
    x_i = x_g / sum_x_g

    ln_x_i = np.log(x_i)

    _lambda = x_g * (np.log(x) * sum_x_g - np.sum(h *
                     x_g * np.log(x))) / (sum_x_g ** 2)
    f = 1 + np.sum(ln_x_i * h) - np.sum(x_i * ln_x_i * h)
    f_prime = np.sum(_lambda * h * (sum_x_g / x_g - ln_x_i - 1))

    return f / f_prime


def weibullMleHist(ax, h):
    eps = 0.01  # precision of Newton-Raphson method's solution
    shape = 0.1  # initial value of gamma parameter

    h = h / np.sum(h)
    shape_next = shape - weibullNewtonHist(shape, ax, h)
    n_iteration = 0

    while np.abs(shape_next - shape) > eps:
        if np.isnan(shape_next) or np.isinf(shape_next) or shape_next > 20 or n_iteration > 30:
            break

        if shape_next <= 0:
            shape_next = 0.000001
            break

        shape = shape_next
        shape_next = shape - weibullNewtonHist(shape, ax, h)

        n_iteration += 1

    shape = shape_next
    scale = np.power(np.sum((np.power(ax, shape) * h)), 1/shape)

    return scale, shape


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
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


def local_cov(e, sigma):
    break_off_sigma = 3.0
    filter_size = np.round(break_off_sigma * sigma)
    h = matlab_style_gauss2D((filter_size, filter_size), sigma)

    h_shape = h.shape[0]
    e_shape = (1,1, e.shape[0], e.shape[1])
    e = e.reshape(e_shape)
    h = torch.tensor(h, device='cuda', dtype=torch.float64).reshape((1,1,h_shape,h_shape)).float()

    # term1 = cv2.filter2D(e_n**2, -1, h_n, borderType=cv2.BORDER_REPLICATE)
    term1_t = torch.nn.functional.conv2d(e**2, h, padding='same')
    # term2 = cv2.filter2D(e_n, -1, h_n, borderType=cv2.BORDER_REPLICATE)**2
    term2_t = torch.nn.functional.conv2d(e, h, padding='same')**2

    term1_t = term1_t.squeeze()
    term2_t = term2_t.squeeze()



    # local_std = np.sqrt(np.maximum(term1 - term2, np.zeros(term1.shape)))
    local_std_t = torch.sqrt(torch.maximum(term1_t - term2_t, torch.zeros(term1_t.shape, device='cuda')))

    # local_mean = cv2.filter2D(
        # e_n, -1, h_n, borderType=cv2.BORDER_REPLICATE) + np.finfo(float).tiny

    local_mean_t = torch.nn.functional.conv2d(e, h.reshape((1,1,h_shape,h_shape)), padding='same').squeeze() + torch.finfo(float).tiny

    # return local_std / local_mean
    return local_std_t / local_mean_t


def create_hist(data, h_bins):

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


def rgb2e(im):
    # convert RGB to e, el, ell

    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    # JMG: slightly different values than in PAMI 2001 paper;
    # simply assuming correctly white balanced camera
    E = (0.3*R + 0.58*G + 0.11*B) / 255.0
    El = (0.25*R + 0.25*G - 0.5*B) / 255.0
    Ell = (0.5*R - 0.5*G) / 255.0

    # # As in original in PAMI 2001 paper
    # E   = (0.06*R + 0.63*G + 0.27*B ) / 255.0
    # El  = (0.3*R  + 0.04*G - 0.35*B ) / 255.0
    # Ell = (0.34*R - 0.6*G+0.17*B) / 255.0

    return E, El, Ell


def conv2padded(varargin):
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
        h = torch.reshape(h, (1, 1, vertical, horizontal))

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
    x_padded = torch.Tensor(np.pad(x.cpu(), pad_width=[(top, bottom), (left, right)], mode='edge'))

    def conv2(v1, v2, m, mode='same'):
        """
        Two-dimensional convolution of matrix m by vectors v1 and v2

        First convolves each column of 'm' with the vector 'v1'
        and then it convolves each row of the result with the vector 'v2'.

        from https://stackoverflow.com/questions/24231285/is-there-a-python-equivalent-to-matlabs-conv2h1-h2-a-same

        """
        tmp = np.apply_along_axis(torch.conv2d, 0, m, v1, mode)
        return np.apply_along_axis(torch.conv2d, 1, tmp, v2, mode)

    if x.ndim == 2:
        x = x.reshape((1,1,x.shape[0], x.shape[1]))
        x_padded = x_padded.reshape((1,1,x_padded.shape[0], x_padded.shape[1]))

    for p in range(x.shape[1]):
        if len(varargin) == 2:
            ans2 = torch.nn.functional.conv2d(x_padded.to(torch.device('cuda')), h, padding='valid')
            x[:, p, :] = ans2.squeeze()
        else:
            x[:, :, p] = conv2(h1, h2, x_padded[:, :, p], mode='valid')

    return x


def filter_lgn(im, sigma):
    break_off_sigma = 3
    filter_size = break_off_sigma * sigma
    x = torch.tensor([i for i in range(-filter_size, filter_size+1)], device='cuda')

    gauss = 1 / (np.sqrt(2*np.pi) * sigma) * \
        torch.exp((x**2) / (-2 * sigma * sigma))
    Gx = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
    Gx = Gx - sum(Gx) / len(x)
    Gx = Gx / sum(0.5 * x * x * Gx)

    Gy = (x**2 / np.power(sigma, 4) - 1/sigma**2) * gauss
    Gy = Gy - sum(Gy) / len(x)
    Gy = Gy / sum(0.5 * x * x * Gy)

    Gx = Gx.to(torch.device('cuda'))
    Gy = Gy.to(torch.device('cuda'))

    if im.shape[2] == 1:
        im = im / torch.max(im)
        Ex = conv2padded((deepcopy(im), Gx))
        Ey = conv2padded((deepcopy(im), Gy.conj()))
        e = torch.sqrt(Ex**2 + Ey**2)
        el = []
        ell = []

    else:
        e, el, ell = rgb2e(im)

        # im = e
        Ex = conv2padded((deepcopy(e), Gx))
        Ey = conv2padded((deepcopy(e), Gy.conj()))
        e = torch.sqrt(Ex**2 + Ey**2).squeeze()

        # im = el
        Elx = conv2padded((deepcopy(el), Gx))
        Ely = conv2padded((deepcopy(el), Gy.conj()))
        el = torch.sqrt(Elx**2 + Ely**2).squeeze()

        # im = ell
        Ellx = conv2padded((deepcopy(ell), Gx))
        Elly = conv2padded((deepcopy(ell), Gy.conj()))
        ell = torch.sqrt(Ellx**2 + Elly**2).squeeze()

    return e, el, ell


def lgn_statistics_cuda(im, file_name:str, threshold_lgn, viewing_dist=1, dot_pitch=0.00035, fov_beta=1.5, fov_gamma=5, 
                    verbose: bool = False, compute_extra_statistics: bool = False, crop_masks: list = [], force_recompute:bool=False):

    result_manager = ResultManager(root='/home/niklas/projects/lgnpy/cache')

    print(f"Computing LGN statistics for {file_name}")
    # Check if file exists
    file_name = f"results_{file_name}.npy"
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

    imsize = im.shape[:2]

    #######################################################
    # Set parameters for field of view
    #######################################################

    fovx = round(imsize[1]/2)          # x-pixel loc. of fovea center
    fovy = round(imsize[0]/2)          # y-pixel loc. of fovea center
    # ex and ey are the x- and y- offsets of each pixel compared to
    # the point of focus (fovx,fovy) in pixels.
    ex, ey = np.meshgrid(np.arange(start=-fovx+1, stop=imsize[1]-fovx+1),
                        np.arange(start=-fovy+1, stop=imsize[0]-fovy+1))
    # eradius is the radial distance between each point and the point
    # of gaze.  This is in meters.
    eradius = dot_pitch * np.sqrt(ex**2+ey**2)
    #del ex, ey
    # calculate ec, the eccentricity from the foveal center, for each
    # point in the image.  ec is in degrees.
    ec = 180*np.arctan(eradius / viewing_dist)/np.pi
    # select the pixels that fall within the input visual field of view
    imfovbeta = (ec < fov_beta)
    imfovgamma = (ec < fov_gamma)


    # (color_channels, (full+boxes), center-peripherie)
    ce = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    sc = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    # ce = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    # sc = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    beta = [] # TODO also change beta, gamma to the center+peripherie computation
    gamma = []
    # beta = np.zeros((im.shape[-1], 1+len(crop_masks), 2))
    # gamma = np.zeros((im.shape[-1], 1+len(crop_masks), 2))

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

        for iteration_index, sigma_iterations in enumerate(np.array([[48, 24, 12, 6, 3], [64, 32, 16, 8, 4]])):
            for _, sigma in enumerate(sigma_iterations):
                if verbose:
                    print(f"Sigma: {sigma}")

                if verbose:
                    print('Interpolate')
                sigmas = np.array([1, 2, 4, 8, 16, 32])
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
                o1, o2, o3 = filter_lgn(im, sigma)

                if verbose:
                    print("Local COV 1")
                s1 = local_cov(o1, sigma)
                s1 = s1.cpu().numpy()

                o1 = o1.cpu().numpy()


                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    e1 = ((o1 * np.max(o1)) / (o1 + np.max(o1) * s1))
                minm1 = e1 - t1
                index1 = (minm1 > 0.0000001)
                if iteration_index == 0:
                    par1[index1] = minm1[index1]
                elif iteration_index == 1:
                    mag1[index1] = minm1[index1]

                if IMTYPE == 2:
                    if verbose:
                        print("Local COV 2")
                    s2 = local_cov(o2, sigma)
                    s2 = s2.cpu().numpy()

                    o2 = o2.cpu().numpy()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        e2 = ((o2 * np.max(o2)) / (o2 + np.max(o2) * s2))
                    minm2 = e2 - t2
                    index2 = (minm2 > 0.0000001)
                    if iteration_index == 0:
                        par2[index2] = minm2[index2]
                    elif iteration_index == 1:
                        mag2[index2] = minm2[index2]

                    if verbose:
                        print("Local COV 3")
                    s3 = local_cov(o3, sigma)
                    s3 = s3.cpu().numpy()

                    o3 = o3.cpu().numpy()

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        e3 = ((o3 * np.max(o3)) / (o3 + np.max(o3) * s3))
                    minm3 = e3 - t3
                    index3 = (minm3 > 0.0000001)
                    if iteration_index == 0:
                        par3[index3] = minm3[index3]
                    elif iteration_index == 1:
                        mag3[index3] = minm3[index3]

    
        results = np.array((par1, par2, par3, mag1, mag2, mag3))
        result_manager.save_result(result=results, filename=file_name, overwrite=True)
        print('Done saving')
        del results
    else:
        par1, par2, par3, mag1, mag2, mag3 = results
    ##############
    # Compute Feature Energy and Spatial Coherence
    ##############

    if verbose:
        print("Compute CE")

    magnitude = np.abs(par1[imfovbeta])
    # magnitude = np.abs(par1[~imfovbeta]) # Here we can select the pixels that lie OUTSIDE the fovea and compute SC/CE on those instead
    # Full scene, red/gray
    # ce.append(np.mean(magnitude))
    ce[0, 0, 0] = np.mean(magnitude)
    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        # ce.append(np.mean(magnitude))
        ce[1, 0, 0] = np.mean(magnitude)
        magnitude = np.abs(par3[imfovbeta])
        # ce.append(np.mean(magnitude))
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
            box_center = np.mean(np.abs(par1[mask]))
            ce[0, mask_index+1, 0] = box_center
            box_peri = np.mean(np.abs(par1[~mask]))
            ce[0, mask_index+1, 1] = box_peri

            if IMTYPE == 2:
                box_center = np.mean(np.abs(par2[mask]))
                ce[1, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par2[~mask]))
                ce[1, mask_index+1, 1] = box_peri
                box_center = np.mean(np.abs(par3[mask]))
                ce[2, mask_index+1, 0] = box_center
                box_peri = np.mean(np.abs(par3[~mask]))
                ce[2, mask_index+1, 1] = box_peri

            # ce_extra.append([, np.mean(
            #     np.abs(par2[mask])), np.mean(np.abs(par3[mask]))])

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
            box_center = np.abs(mag1[mask])
            sc[0, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
            box_peri = np.abs(mag1[~mask])
            sc[0, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

            if IMTYPE == 2:
                box_center = np.abs(mag2[mask])
                sc[1, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag2[~mask])
                sc[1, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)
                box_center = np.abs(mag3[mask])
                sc[2, mask_index+1, 0] = np.mean(box_center) / np.std(box_center)
                box_peri = np.abs(mag3[~mask])
                sc[2, mask_index+1, 1] = np.mean(box_peri) / np.std(box_peri)

    #################
    # Compute Weibull parameters beta and gamma
    #################

    if verbose:
        print("Compute Weibull parameters beta")

    n_bins = 1000
    magnitude = np.abs(par1[imfovbeta])
    ax, h = create_hist(magnitude, n_bins)
    beta.append(weibullMleHist(ax, h)[0])

    if IMTYPE == 2:
        magnitude = np.abs(par2[imfovbeta])
        ax, h = create_hist(magnitude, n_bins)
        beta.append(weibullMleHist(ax, h)[0])

        magnitude = np.abs(par3[imfovbeta])
        ax, h = create_hist(magnitude, n_bins)
        beta.append(weibullMleHist(ax, h)[0])

    if verbose:
        print("Compute Weibull parameters gamma")
    magnitude = np.abs(mag1[imfovgamma])
    ax, h = create_hist(magnitude, n_bins)
    gamma.append(weibullMleHist(ax, h)[1])

    if IMTYPE == 2:
        magnitude = np.abs(mag2[imfovgamma])
        ax, h = create_hist(magnitude, n_bins)
        gamma.append(weibullMleHist(ax, h)[1])

        magnitude = np.abs(mag3[imfovgamma])
        ax, h = create_hist(magnitude, n_bins)
        gamma.append(weibullMleHist(ax, h)[1])

    return (ce, sc, beta, gamma)
