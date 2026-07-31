
import numpy as np

IMG_SIZE = 32


def _gaussian_blob(size, cx, cy, sigma, amplitude):
    y, x = np.mgrid[0:size, 0:size]
    return amplitude * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2)))


def make_identity_params(rng, identity_id):
   
    return {
        "id": identity_id,
        "eye_l": (rng.uniform(9, 12), rng.uniform(11, 14)),
        "eye_r": (rng.uniform(20, 23), rng.uniform(11, 14)),
        "nose": (rng.uniform(15, 17), rng.uniform(16, 19)),
        "mouth": (rng.uniform(15, 17), rng.uniform(23, 26)),
        "eye_sigma": rng.uniform(1.5, 2.5),
        "nose_sigma": rng.uniform(2.0, 3.0),
        "mouth_sigma": rng.uniform(2.5, 4.0),
        "eye_amp": rng.uniform(0.5, 0.8),
        "nose_amp": rng.uniform(0.3, 0.5),
        "mouth_amp": rng.uniform(0.4, 0.7),
        "base_brightness": rng.uniform(0.25, 0.35),
    }


def render_sample(rng, params, position_jitter=0.6, sensor_noise=0.03, lighting_jitter=0.05):
    
    img = np.full((IMG_SIZE, IMG_SIZE), params["base_brightness"], dtype=np.float64)

    lighting = 1.0 + rng.uniform(-lighting_jitter, lighting_jitter)

    for key, sigma_key, amp_key in [
        ("eye_l", "eye_sigma", "eye_amp"),
        ("eye_r", "eye_sigma", "eye_amp"),
        ("nose", "nose_sigma", "nose_amp"),
        ("mouth", "mouth_sigma", "mouth_amp"),
    ]:
        cx, cy = params[key]
        cx = cx + rng.uniform(-position_jitter, position_jitter)
        cy = cy + rng.uniform(-position_jitter, position_jitter)
        img += _gaussian_blob(IMG_SIZE, cx, cy, params[sigma_key], params[amp_key] * lighting)

    img += rng.normal(0, sensor_noise, size=img.shape)
    return np.clip(img, 0, 1)


def generate_dataset(num_identities, samples_per_identity, seed=0,
                      position_jitter=0.6, sensor_noise=0.03):
    
    identity_rng_seed_base = 1_000_003  
    identities = [
        make_identity_params(np.random.default_rng(identity_rng_seed_base + i), i)
        for i in range(num_identities)
    ]

    session_rng = np.random.default_rng(seed)
    X, y = [], []
    for params in identities:
        for _ in range(samples_per_identity):
            img = render_sample(session_rng, params, position_jitter, sensor_noise)
            X.append(img.flatten())
            y.append(params["id"])
    return np.array(X), np.array(y), identities
