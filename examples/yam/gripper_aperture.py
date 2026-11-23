"""Convert predicted YAM gripper apertures to commands in [0, 1]."""
import numpy as np


# These bounds apply to raw model output, before clipping the motor command.
_PREDICTION_BOUNDS = {
    "flux3-action": (-0.05, 1.06),
    "longwam": (-0.06, 1.06),
    "molmoact2": (-0.15, 1.15),
}


def clip_predicted_aperture(gripper, model_label, *, tolerance=None):
    if tolerance is None:
        lower, upper = _PREDICTION_BOUNDS.get(model_label, (-0.05, 1.05))
    else:
        lower, upper = -tolerance, 1.0 + tolerance
    if not np.isfinite(gripper).all() or np.any((gripper < lower) | (gripper > upper)):
        raise ValueError(
            f"{model_label} returned gripper targets ranging from {gripper.min():.4f} to "
            f"{gripper.max():.4f}, outside the accepted aperture range [{lower:.2f}, {upper:.2f}]"
        )
    return np.clip(gripper, 0, 1)
