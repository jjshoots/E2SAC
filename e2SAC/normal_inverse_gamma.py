import math

import torch
import torch.distributions as D
import torch.nn.functional as F

"""
Normal Inverse Gamma Distribution for evidential uncertainty learning adapted for Torch:
    Deep Evidential Regression, Amini et. al.
        https://arxiv.org/pdf/1910.02600.pdf
        https://www.youtube.com/watch?v=toTcf7tZK8c
        https://github.com/aamini/evidential-deep-learning
"""


def NormalInvGamma(gamma, nu, alpha, beta):
    """
    Normal Inverse Gamma Distribution
    """
    assert torch.all(nu > 0.0), f"nu must be more than zero, min value is {nu.min()}"
    assert torch.all(
        alpha > 1.0
    ), f"alpha must be more than one, min value is {alpha.min()}"
    assert torch.all(
        beta > 0.0
    ), f"beta must be more than zero, min value is {beta.min()}"

    InvGamma = D.transformed_distribution.TransformedDistribution(
        D.Gamma(alpha, beta),
        D.transforms.PowerTransform(torch.tensor(-1.0).to(alpha.device)),
    )

    var = InvGamma.rsample()
    mu = D.Normal(gamma, torch.sqrt(beta / (alpha - 1) / nu)).rsample()

    return D.Normal(mu, torch.sqrt(var))


def ShrunkenNormalInvGamma(gamma, nu, alpha, beta, clamp_mean=None, clamp_var=None):
    """
    Normal Inverse Gamma Distribution
    """
    assert torch.all(
        alpha > 1.0
    ), f"alpha must be more than one, min value is {alpha.min()}"
    assert torch.all(
        beta > 0.0
    ), f"beta must be more than zero, min value is {beta.min()}"

    var = beta / (alpha - 1)

    if clamp_mean is not None:
        if isinstance(clamp_mean, list):
            gamma = F.hardtanh(gamma, clamp_mean[0], clamp_mean[1])
        else:
            gamma = F.hardtanh(gamma, -clamp_mean, clamp_mean)

    if clamp_var is not None:
        if isinstance(clamp_var, list):
            var = F.hardtanh(var, clamp_var[0], clamp_var[1])
        else:
            var = F.hardtanh(var, -clamp_var, clamp_var)

    return D.Normal(gamma, torch.sqrt(var))


def NIG_epistemic(gamma, nu, alpha, beta):
    """
    calculates the epistemic uncertainty of a distribution
    the value is effectively expectation of sigma square
    """
    assert torch.all(nu > 0.0), f"nu must be more than zero, min value is {nu.min()}"
    assert torch.all(
        alpha > 1.0
    ), f"alpha must be more than one, min value is {alpha.min()}"
    assert torch.all(
        beta > 0.0
    ), f"beta must be more than zero, min value is {beta.min()}"

    return torch.sqrt(beta / (alpha - 1) / nu)


def NIG_aleatoric(gamma, nu, alpha, beta):
    """
    calculates the aleatoric uncertainty of a distribution
    the value is effectively expectation of sigma square
    """
    assert torch.all(nu > 0.0), f"nu must be more than zero, min value is {nu.min()}"
    assert torch.all(
        alpha > 1.0
    ), f"alpha must be more than one, min value is {alpha.min()}"
    assert torch.all(
        beta > 0.0
    ), f"beta must be more than zero, min value is {beta.min()}"

    return torch.sqrt(beta / (alpha - 1))


def NIG_NLL(label, gamma, nu, alpha, beta, reduce=True):
    """
    Negative Log Likelihood loss between label and predicted output
    """
    twoBlambda = 2 * beta * (1 + nu)

    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(twoBlambda)
        + (alpha + 0.5) * torch.log(nu * (label - gamma) ** 2 + twoBlambda)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return torch.mean(nll) if reduce else nll


def NIG_reg(label, gamma, nu, alpha, beta, reduce=True):
    """
    Regularizer for for NIG distribution, scale the output of this by ~0.01
    """
    loss = torch.abs(gamma - label) * (2 * nu + alpha)
    return torch.mean(loss) if reduce else loss
