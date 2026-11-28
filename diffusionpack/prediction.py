from .restore import restore_loss, restore_sample


def prediction_loss(model, future, history):
    return restore_loss(model, future, history)


def prediction_sample(model, history, *, progress=False, sampler="ddpm",
                      sampling_steps=None, eta=0.0):
    return restore_sample(model, history, clip_denoised=False, progress=progress,
                          sampler=sampler, sampling_steps=sampling_steps, eta=eta)
