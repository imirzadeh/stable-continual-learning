""" Top-level module for hessian eigenvec computation
This library is cited in our paper.
"""
from external_libs.hessian_eigenthings.power_iter import power_iteration, deflated_power_iteration
from external_libs.hessian_eigenthings.lanczos import lanczos
from external_libs.hessian_eigenthings.hvp_operator import HVPOperator, compute_hessian_eigenthings

__all__ = [
    "power_iteration",
    "deflated_power_iteration",
    "lanczos",
    "HVPOperator",
    "compute_hessian_eigenthings",
]

name = "hessian_eigenthings"
