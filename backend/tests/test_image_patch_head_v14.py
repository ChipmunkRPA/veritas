"""Source geometry, supervision, optimization and independent arithmetic guards."""
import numpy as np
import pytest
import torch

from scripts import image_patch_head_v14 as method
from scripts import review_image_patch_head_v14 as review
from scripts import run_image_patch_head_v14 as study


def test_constant_rgb_normalization_has_the_right_channels_and_geometry():
    rgb = np.full((51, 73, 3), [17, 101, 239], np.uint8)
    value = method.normalize(rgb)
    assert value.shape == (1, 3, 448, 448) and value.dtype == np.float32
    expected = (np.array([17, 101, 239], np.float32) / 255 - method.MEAN) / method.STD
    assert np.array_equal(value[0, :, 71, 93], expected)


@pytest.mark.parametrize('height,width', [(32, 32), (33, 65), (71, 97)])
def test_every_odd_source_pixel_has_exactly_one_soft_label_cell(height, width):
    y, x = np.indices((height, width))
    truth = (x + 3 * y) % 7 == 0
    pooled = method.pool_truth(truth)
    assert np.array_equal(pooled, review.independent_truth(truth))
    pixel_cells = y * 0 + (y * 32 // height) * 32 + x * 32 // width
    counts = np.bincount(pixel_cells.ravel(), minlength=1024)
    assert counts.sum() == truth.size and (counts > 0).all()
    assert abs(np.dot(counts, pooled.ravel().astype(np.float64)) - truth.sum()) < .0001


def test_stratified_sampling_is_deterministic_and_preserves_partial_targets():
    target = np.zeros((32, 32), np.float32)
    target[5, 8] = .125
    indices = method.indices(target, 'authored-source')
    assert np.array_equal(indices, review.independent_indices(target, 'authored-source'))
    assert np.count_nonzero(target.ravel()[indices] == .125) == 32
    assert np.count_nonzero(target.ravel()[indices] == 0) == 32


def test_actual_small_head_learns_and_refuses_missing_class_supervision():
    torch.set_num_threads(2)
    rng = np.random.default_rng(4)
    labels = np.repeat([0., 1.], 8).astype(np.float32)
    samples = rng.normal(0, .1, (16, 64, 2)).astype(np.float32)
    samples[..., 0] += (labels * 2 - 1)[:, None]
    means = np.zeros((16, 2), np.float32)
    targets = np.repeat(labels[:, None], 64, axis=1)
    head, receipt = method.train(samples, means, targets)
    assert receipt['losses'][-1] < .6 * receipt['losses'][0]
    with torch.inference_mode():
        x = torch.from_numpy(samples.reshape(-1, 2))
        predicted = head(torch.cat([x, x], dim=1)).flatten().numpy() > 0
    assert np.mean(predicted == targets.ravel()) >= .95
    with pytest.raises(ValueError, match='Both soft supervision classes'):
        method.train(samples, means, np.zeros_like(targets))


def test_independent_float64_head_matches_native_dense_relu_sigmoid_and_median():
    torch.manual_seed(11)
    head = method.new_head().eval()
    rng = np.random.default_rng(8)
    values = rng.normal(0, .2, (1024, 2304)).astype(np.float32)
    average = values.mean(axis=0)
    native = method.response(head, values, average)
    weights = {k: v.detach().numpy().astype(np.float64) for k, v in head.state_dict().items()}
    independent = review.numpy_scores(values, average, weights)
    assert np.abs(native - independent).max() < 1e-6


def test_four_diagonally_connected_cells_survive_and_strict_point_ties_do_not():
    scores = np.full((32, 32), .5, np.float32)
    for k in range(4):
        scores[k, k] = .75
    expected = review.independent_mask(scores, .5, 79, 47)
    native = method.selected(scores, .5, 79, 47)
    assert np.array_equal(native, expected) and native.any()
    scores[3, 3] = .5
    assert not method.selected(scores, .5, 79, 47).any()


def test_point_requires_both_genuine_variants_from_all_sixty_sources():
    records = [{'group': str(k), 'role': 'point', 'label': 0, 'maximum': k / 60}
               for k in range(60) for _ in range(2)]
    point = study.point(records)
    assert point['value'] == 56 / 60
    records[0]['label'] = 1
    with pytest.raises(ValueError, match='Only genuine point-role'):
        study.point(records)


def test_coordinate_review_counts_false_highlights_without_ai_authorship():
    truth, mask = np.zeros((33, 65), bool), np.zeros((33, 65), bool)
    mask[7:13, 8:21] = True
    got = review.metrics(mask, truth)
    assert got['false_positive_pixels'] == 78 and got['flag'] and got['iou'] == 0
    assert 'confidence' not in got


def test_renaming_a_source_does_not_allow_its_bytes_to_cross_roles():
    entries = [{'role': role, 'source': {'group': name, 'source_sha256': 'same-parent',
                'input_sha256': name + '-input', 'rgb_sha256': name + '-rgb'}}
               for role, name in [('fit', 'first'), ('diagnostic', 'renamed')]]
    with pytest.raises(ValueError, match='bytes cannot cross source roles'):
        study.source_roles(entries)
