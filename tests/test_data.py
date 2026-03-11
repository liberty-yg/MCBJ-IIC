from __future__ import annotations

import numpy as np

from mcbj_iic.data import infer_crop_size, preprocess_matlab_dataset, preprocess_raw_traces, resolve_mat_dataset_path


def _make_trace(length: int, label_offset: float = 0.0):
    displacement = np.linspace(-0.3, 2.0, length)
    conductance = np.linspace(0.0 + label_offset, -5.0 + label_offset, length)
    return np.column_stack([displacement, conductance])


def test_infer_crop_size_ignores_negative_displacement_region():
    raw = np.array([_make_trace(10), _make_trace(14)], dtype=object)
    crop_size = infer_crop_size(raw)
    assert crop_size > 0
    assert isinstance(crop_size, int)


def test_preprocess_raw_traces_returns_fixed_length_array():
    raw = np.array([_make_trace(12), _make_trace(18)], dtype=object)
    processed = preprocess_raw_traces(raw, crop_size=20, conductance_floor=-5.5)
    assert processed.shape == (2, 20)
    assert np.isfinite(processed).all()


def test_preprocess_matlab_dataset_zero_bases_labels_and_adds_channel_dim():
    raw = np.array([_make_trace(12), _make_trace(18), _make_trace(16)], dtype=object)
    labels = np.array([[1], [2], [7]])
    matlab_dict = {"Data": raw, "Labels": labels}

    bundle = preprocess_matlab_dataset(matlab_dict, crop_size=24, shuffle=False)

    assert bundle.x.shape == (3, 24, 1)
    assert bundle.y.tolist() == [0, 1, 6]
    assert bundle.crop_size == 24


def test_resolve_mat_dataset_path_prefers_data_mat(tmp_path):
    dataset = tmp_path / "Data.mat"
    dataset.write_text("placeholder", encoding="utf-8")

    resolved = resolve_mat_dataset_path(search_roots=[tmp_path])
    assert resolved == dataset.resolve()


def test_resolve_mat_dataset_path_requires_explicit_choice_when_multiple_files_exist(tmp_path):
    (tmp_path / "first.mat").write_text("a", encoding="utf-8")
    (tmp_path / "second.mat").write_text("b", encoding="utf-8")

    try:
        resolve_mat_dataset_path(search_roots=[tmp_path])
    except FileNotFoundError as exc:
        assert "Multiple local .mat files" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when multiple .mat files are present")
