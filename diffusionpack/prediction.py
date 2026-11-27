from .restore import restore_loss, restore_sample
from .scrd.prediction import scrd_prediction_loss, scrd_prediction_sample


def prediction_loss(model, future, history):
    return restore_loss(model, future, history)


def prediction_sample(model, history, *, progress=False):
    return restore_sample(model, history, clip_denoised=False, progress=progress)
