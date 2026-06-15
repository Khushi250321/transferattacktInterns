import os
import uuid
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
from PIL import Image
from deepface import DeepFace

ATTACKER_MODELS = {
    'Facenet512': (160, 160),
    'ArcFace': (112, 112),
    'GhostFaceNet': (112, 112),
    'VGG-Face': (224, 224),
}

VICTIM_MODELS = ['Facenet512', 'ArcFace', 'GhostFaceNet', 'VGG-Face', 'IR152']

ALL_ATTACKS = [
    'PGD',
    'MI_FGSM',
    'TI_FGSM',
    'SI_NI_FGSM',
    'MI_ADMIX_DI_TI',
    'BPA_CNN',
]

ATTACK_COLS = {
    'PGD': 'pgd_path',
    'MI_FGSM': 'mi_fgsm_path',
    'TI_FGSM': 'ti_fgsm_path',
    'SI_NI_FGSM': 'si_ni_fgsm_path',
    'MI_ADMIX_DI_TI': 'mi_admix_di_ti_path',
    'BPA_CNN': 'bpa_cnn_path',
}

EPSILON = 0.062
NUM_ITER = 5
DECAY = 1.0


def configure_cpu_runtime(tf_threads: int = 1) -> None:
    try:
        tf.config.set_visible_devices([], 'GPU')
    except Exception:
        pass
    try:
        tf.config.threading.set_intra_op_parallelism_threads(tf_threads)
        tf.config.threading.set_inter_op_parallelism_threads(tf_threads)
    except Exception:
        pass


def resolve_image_path(path: str, dataset_root: str) -> str:
    value = str(path)
    if os.path.exists(value):
        return value
    marker = 'dataset_extractedfaces/'
    if marker in value:
        rel = value.split(marker, 1)[1]
        candidate = os.path.join(dataset_root, rel)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(dataset_root, value.lstrip('/'))


def load_and_preprocess(path: str, input_size):
    img = Image.open(path).convert('RGB').resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    return (arr - 0.5) * 2.0


def denormalize(x: np.ndarray) -> np.ndarray:
    x = (x + 1.0) / 2.0
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def compute_embedding(model, x):
    out = model(x, training=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return tf.nn.l2_normalize(out, axis=1)


def attack_loss(cos, attack_type: str):
    return tf.reduce_mean(cos if str(attack_type).strip().lower() == 'impersonation_attack' else (1 - cos))


def save_adv(img_uint8: np.ndarray, attack_name: str, src: str, tgt: str, attack_type: str, model_name: str, row_id: int, adv_root: str) -> str:
    out_dir = Path(adv_root) / model_name / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)
    s = Path(src).stem.replace(' ', '_')
    t = Path(tgt).stem.replace(' ', '_')
    rand = uuid.uuid4().hex[:8]
    name = f'adv_r{row_id}_{s}_to_{t}_{attack_type}_{rand}.png'
    path = out_dir / name
    Image.fromarray(img_uint8).save(path)
    return str(path.resolve())


def gaussian_kernel(k=15, sigma=3.0, ch=3):
    x = tf.range(-k // 2 + 1, k // 2 + 1, dtype=tf.float32)
    g = tf.exp(-tf.square(x) / (2 * sigma**2))
    g /= tf.reduce_sum(g)
    kernel = tf.tensordot(g, g, axes=0)
    kernel = kernel[:, :, None, None]
    return tf.tile(kernel, [1, 1, ch, 1])


def input_diversity(x, input_size, prob=0.7):
    if tf.random.uniform([]) > prob:
        return x
    img_size = input_size[0]
    rnd = tf.random.uniform([], int(0.9 * img_size), img_size, dtype=tf.int32)
    x_resized = tf.image.resize(x, (rnd, rnd))
    pad_total = img_size - rnd
    pad_top = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_bottom = pad_total - pad_top
    pad_left = tf.random.uniform([], 0, pad_total + 1, dtype=tf.int32)
    pad_right = pad_total - pad_left
    x_padded = tf.pad(x_resized, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
    return tf.image.resize(x_padded, input_size)


def pgd_attack(model, x, tgt_emb, attack_type, random_start=True):
    if random_start:
        noise = tf.random.uniform(tf.shape(x), minval=-EPSILON, maxval=EPSILON, dtype=x.dtype)
        adv = tf.clip_by_value(x + noise, -1.0, 1.0)
    else:
        adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def ti_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    alpha = EPSILON / NUM_ITER
    kernel = gaussian_kernel()
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        adv = adv + alpha * tf.sign(grad)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def si_ni_fgsm(model, x, tgt_emb, attack_type):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    scales = (1.0, 0.5, 0.25, 0.125, 0.0625)
    for _ in range(NUM_ITER):
        nes = adv + DECAY * alpha * g
        grad_sum = tf.zeros_like(x)
        for s in scales:
            with tf.GradientTape() as tape:
                tape.watch(nes)
                emb = compute_embedding(model, nes * s)
                cos = tf.reduce_sum(emb * tgt_emb, axis=1)
                loss = attack_loss(cos, attack_type)
            grad_sum += tape.gradient(loss, nes)
        grad = grad_sum / len(scales)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv


def mi_admix_di_ti(model, x, tgt_emb, attack_type, pool_imgs, input_size):
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel()
    n_pool = tf.shape(pool_imgs)[0]
    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            idx = tf.random.uniform([3], 0, n_pool, dtype=tf.int32)
            others = tf.gather(pool_imgs, idx)
            adv_rep = tf.repeat(adv, 3, axis=0)
            mixed = adv_rep + 0.2 * (others - adv_rep)
            batch = input_diversity(mixed, input_size)
            emb = compute_embedding(model, batch)
            tgt_rep = tf.repeat(tgt_emb, 3, axis=0)
            cos = tf.reduce_sum(emb * tgt_rep, axis=1)
            loss = attack_loss(cos, attack_type)
        grad = tape.gradient(loss, adv)
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
    return adv

# Student-contributed attack integration:
# BPA_CNN by Om Singh Rawat (IIT Delhi)
# Paper basis: Rethinking the Backward Propagation for Adversarial Transferability
# (NeurIPS 2023)
def bpa_cnn(model, x, tgt_emb, attack_type):
    """BPA-CNN: Backward Propagation Attack adapted for CNN face models.

    BPA (NeurIPS 2023) improves adversarial transferability by replacing
    sharp backward operations (ReLU, MaxPool) with smooth alternatives
    (SiLU derivative, softmax-weighted pooling).

    Since we cannot modify internal layers of pre-trained DeepFace models,
    we apply BPA's two key gradient-smoothing principles at the input level:
      1. SiLU-derivative scaling  – counteracts ReLU binary gradient masking
      2. Gaussian spatial smoothing – counteracts MaxPool gradient concentration
    """
    adv = tf.identity(x)
    g = tf.zeros_like(x)
    alpha = EPSILON / NUM_ITER
    tgt_emb = tf.nn.l2_normalize(tgt_emb, axis=1)
    kernel = gaussian_kernel(k=5, sigma=1.0)
    temperature = 3.0

    for _ in range(NUM_ITER):
        with tf.GradientTape() as tape:
            tape.watch(adv)
            emb = compute_embedding(model, adv)
            cos = tf.reduce_sum(emb * tgt_emb, axis=1)
            loss = attack_loss(cos, attack_type)

        grad = tape.gradient(loss, adv)

        # BPA Step 1: SiLU-inspired gradient smoothing
        # SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        # Applied to the gradient to counteract ReLU's binary masking
        scaled = temperature * grad
        sig = tf.sigmoid(scaled)
        silu_deriv = sig + scaled * sig * (1.0 - sig)
        grad = grad * silu_deriv

        # BPA Step 2: Gaussian spatial smoothing
        # Counteracts MaxPool's winner-take-all gradient concentration
        grad = tf.nn.depthwise_conv2d(grad, kernel, [1, 1, 1, 1], 'SAME')

        # Momentum accumulation (inherited from MI-FGSM base)
        grad = grad / (tf.reduce_mean(tf.abs(grad)) + 1e-8)
        g = DECAY * g + grad
        adv = adv + alpha * tf.sign(g)
        adv = tf.clip_by_value(adv, x - EPSILON, x + EPSILON)
        adv = tf.clip_by_value(adv, -1.0, 1.0)

    return adv


def build_attacker(model_name: str):
    return DeepFace.build_model(model_name).model


def run_attack(attack_name: str, model, src, tgt, attack_type: str, input_size):
    tgt_emb = compute_embedding(model, tgt)
    if attack_name == 'PGD':
        return pgd_attack(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_FGSM':
        return mi_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'TI_FGSM':
        return ti_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'SI_NI_FGSM':
        return si_ni_fgsm(model, src, tgt_emb, attack_type)
    if attack_name == 'MI_ADMIX_DI_TI':
        pool_imgs = tf.concat([src, tgt, src], axis=0)
        return mi_admix_di_ti(model, src, tgt_emb, attack_type, pool_imgs, input_size)
    if attack_name == 'BPA_CNN':
        return bpa_cnn(model, src, tgt_emb, attack_type)
    raise ValueError(f'Unsupported attack: {attack_name}')
