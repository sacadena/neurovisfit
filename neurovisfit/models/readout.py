from neuralpredictors.layers.readouts import MultiReadoutBase
from neuralpredictors.layers.readouts.gaussian import FullGaussian2d


class MultiReadoutFullGaussian2d(MultiReadoutBase):
    _base_readout = FullGaussian2d
