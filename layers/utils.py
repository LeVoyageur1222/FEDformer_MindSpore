import mindspore as ms
import mindspore.ops as ops
import numpy as np
from sympy import Poly, Symbol, legendre, chebyshevt
from scipy.special import eval_legendre
from functools import partial

### 数学工具部分（小波滤波器相关）

def legendreDer(k, x):
    def _legendre(k, x):
        return (2*k+1) * eval_legendre(k, x)
    out = 0
    for i in np.arange(k-1, -1, -2):
        out += _legendre(i, x)
    return out

def phi_(phi_c, x, lb=0, ub=1):
    mask = np.logical_or(x<lb, x>ub).astype(np.float64)
    return np.polynomial.polynomial.Polynomial(phi_c)(x) * (1-mask)

def get_phi_psi(k, base):
    x = Symbol('x')
    phi_coeff = np.zeros((k, k))
    phi_2x_coeff = np.zeros((k, k))

    if base == 'legendre':
        for ki in range(k):
            coeff_ = Poly(legendre(ki, 2*x-1), x).all_coeffs()
            phi_coeff[ki, :ki+1] = np.flip(np.sqrt(2*ki+1) * np.array(coeff_, dtype=np.float64))
            coeff_ = Poly(legendre(ki, 4*x-1), x).all_coeffs()
            phi_2x_coeff[ki, :ki+1] = np.flip(np.sqrt(2) * np.sqrt(2*ki+1) * np.array(coeff_, dtype=np.float64))

        psi1_coeff = np.zeros((k, k))
        psi2_coeff = np.zeros((k, k))

        for ki in range(k):
            psi1_coeff[ki, :] = phi_2x_coeff[ki, :]
            for i in range(k):
                prod_ = np.convolve(phi_2x_coeff[ki, :ki+1], phi_coeff[i, :i+1])
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, np.arange(len(prod_))+1)).sum()
                psi1_coeff[ki, :] -= proj_ * phi_coeff[i, :]
                psi2_coeff[ki, :] -= proj_ * phi_coeff[i, :]
            for j in range(ki):
                prod_ = np.convolve(psi1_coeff[j, :], phi_2x_coeff[ki, :ki+1])
                proj_ = (prod_ * 1/(np.arange(len(prod_))+1) * np.power(0.5, np.arange(len(prod_))+1)).sum()
                psi1_coeff[ki, :] -= proj_ * psi1_coeff[j, :]
                psi2_coeff[ki, :] -= proj_ * psi2_coeff[j, :]

            norm1 = (np.convolve(psi1_coeff[ki, :], psi1_coeff[ki, :]) * 1/(np.arange(2*ki+1)+1) * np.power(0.5, np.arange(2*ki+1)+1)).sum()
            norm2 = (np.convolve(psi2_coeff[ki, :], psi2_coeff[ki, :]) * 1/(np.arange(2*ki+1)+1) * (1 - np.power(0.5, np.arange(2*ki+1)+1))).sum()
            norm_ = np.sqrt(norm1 + norm2)

            psi1_coeff[ki, :] /= norm_
            psi2_coeff[ki, :] /= norm_

        phi = [np.poly1d(np.flip(phi_coeff[i, :])) for i in range(k)]
        psi1 = [np.poly1d(np.flip(psi1_coeff[i, :])) for i in range(k)]
        psi2 = [np.poly1d(np.flip(psi2_coeff[i, :])) for i in range(k)]

    elif base == 'chebyshev':
        raise NotImplementedError('Chebyshev version not implemented for simplicity')

    return phi, psi1, psi2

def get_filter(base, k):
    if base not in ['legendre']:
        raise ValueError(f'Base {base} not supported.')

    x = Symbol('x')
    H0 = np.zeros((k, k))
    H1 = np.zeros((k, k))
    G0 = np.zeros((k, k))
    G1 = np.zeros((k, k))
    PHI0 = np.eye(k)
    PHI1 = np.eye(k)

    phi, psi1, psi2 = get_phi_psi(k, base)
    roots = Poly(legendre(k, 2*x-1)).all_roots()
    x_m = np.array([float(rt.evalf()) for rt in roots])

    wm = 1 / k / legendreDer(k, 2*x_m-1) / eval_legendre(k-1, 2*x_m-1)

    for ki in range(k):
        for kpi in range(k):
            H0[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki](x_m/2) * phi[kpi](x_m)).sum()
            G0[ki, kpi] = 1/np.sqrt(2) * (wm * psi1[ki](x_m/2) * phi[kpi](x_m)).sum()
            H1[ki, kpi] = 1/np.sqrt(2) * (wm * phi[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()
            G1[ki, kpi] = 1/np.sqrt(2) * (wm * psi2[ki]((x_m+1)/2) * phi[kpi](x_m)).sum()

    return H0, H1, G0, G1, PHI0, PHI1

### 简单数据归一化器

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

class GaussianNormalizer(object):
    def __init__(self, x, eps=1e-5):
        self.mean = x.mean()
        self.std = x.std()
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        return x * (self.std + self.eps) + self.mean

class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        mymin = x.min(axis=0)
        mymax = x.max(axis=0)
        self.a = (high - low) / (mymax - mymin)
        self.b = -self.a * mymax + high

    def encode(self, x):
        return self.a * x + self.b

    def decode(self, x):
        return (x - self.b) / self.a

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def abs(self, x, y):
        h = 1.0 / (x.shape[1] - 1)
        norms = (h**(self.d/self.p)) * ops.norm(x - y, ord=self.p, axis=1)
        return norms.mean() if self.size_average else norms.sum()

    def rel(self, x, y):
        diff_norms = ops.norm(x - y, ord=self.p, axis=1)
        y_norms = ops.norm(y, ord=self.p, axis=1)
        rel_norms = diff_norms / y_norms
        return rel_norms.mean() if self.size_average else rel_norms.sum()

    def __call__(self, x, y):
        return self.rel(x, y)
