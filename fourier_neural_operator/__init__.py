import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("fourier_neural_operator").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.13"