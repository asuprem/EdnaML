try:
    from ednaml.metrics.training.Accuracy import TorchAccuracy
except ImportError:
    print('Unable to import EdnaML TorchMetrics.')