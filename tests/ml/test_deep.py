"""Phase 7 tests: neural building blocks and the deep detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

jax = pytest.importorskip("jax")

from cloudsentinel.features import FEATURE_COLUMNS  # noqa: E402
from cloudsentinel.models import available, build, dataset_split, neural  # noqa: E402
from cloudsentinel.models.deep import SEQUENCE_LENGTH, build_sequences  # noqa: E402
from tests.ml.test_models import synthetic_store  # noqa: E402


def normal_matrix(rows: int = 600, features: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, (rows, features)).astype(np.float32)


def test_autoencoder_training_reduces_loss() -> None:
    data = normal_matrix()
    key = jax.random.PRNGKey(0)
    params = neural.init_autoencoder(key, data.shape[1], (16, 8), 4)
    trained, history = neural.train_loop(params, data, neural.autoencoder_loss, epochs=15)

    assert history[-1] < history[0] * 0.8
    assert np.isfinite(history).all()
    del trained


def test_autoencoder_error_is_higher_for_out_of_distribution_rows() -> None:
    data = normal_matrix()
    key = jax.random.PRNGKey(0)
    params = neural.init_autoencoder(key, data.shape[1], (16, 8), 4)
    trained, _ = neural.train_loop(params, data, neural.autoencoder_loss, epochs=25)

    outliers = np.full((20, data.shape[1]), 12.0, dtype=np.float32)
    normal_error = float(np.mean(neural.autoencoder_error(trained, jax.numpy.asarray(data))))
    outlier_error = float(np.mean(neural.autoencoder_error(trained, jax.numpy.asarray(outliers))))

    assert outlier_error > normal_error * 5


def test_vae_training_is_stable() -> None:
    data = normal_matrix()
    params = neural.init_vae(jax.random.PRNGKey(1), data.shape[1], (16, 8), 4)
    trained, history = neural.train_loop(params, data, neural.vae_loss, epochs=15, needs_key=True)
    assert np.isfinite(history).all()
    assert history[-1] < history[0]
    assert np.isfinite(np.asarray(neural.vae_error(trained, jax.numpy.asarray(data)))).all()


def test_gru_encoder_returns_the_final_state() -> None:
    params = neural.init_sequence_autoencoder(jax.random.PRNGKey(2), 5, 12, 4)
    sequences = jax.numpy.asarray(np.random.default_rng(0).normal(size=(7, SEQUENCE_LENGTH, 5)))
    state = neural.gru_encode(params, sequences)
    assert state.shape == (7, 12)
    assert np.isfinite(np.asarray(state)).all()


def test_training_is_reproducible_for_a_given_seed() -> None:
    data = normal_matrix()
    params = neural.init_autoencoder(jax.random.PRNGKey(3), data.shape[1], (12,), 4)
    first, _ = neural.train_loop(params, data, neural.autoencoder_loss, epochs=5, seed=7)
    second, _ = neural.train_loop(params, data, neural.autoencoder_loss, epochs=5, seed=7)
    assert np.allclose(first["encoder"][0]["w"], second["encoder"][0]["w"])


def test_sequences_never_cross_identities_and_never_see_the_future() -> None:
    store = (
        synthetic_store(rows=120)
        .sort_values(["identity_id", "window_start"])
        .reset_index(drop=True)
    )
    matrix = store[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    sequences, _ = build_sequences(store, matrix, length=SEQUENCE_LENGTH)

    assert sequences.shape == (len(store), SEQUENCE_LENGTH, len(FEATURE_COLUMNS))
    # The sequence for row i ends at row i itself — nothing after it is included.
    for row in (0, 5, len(store) - 1):
        assert np.allclose(sequences[row, -1], matrix[row])

    # The first window of an identity is left-padded with itself, not with the
    # previous identity's data.
    first_rows = store.groupby("identity_id", observed=True).head(1).index
    for row in first_rows:
        assert np.allclose(sequences[row, 0], matrix[row])


@pytest.mark.parametrize("name", ["autoencoder", "vae"])
def test_deep_detectors_return_bounded_scores(name: str) -> None:
    store = synthetic_store()
    detector = build(name)
    detector.params["epochs"] = 5
    detector.fit(store[list(FEATURE_COLUMNS)])
    scores = detector.score(store[list(FEATURE_COLUMNS)])

    assert len(scores) == len(store)
    assert np.all((scores >= 0) & (scores <= 1))


def test_sequence_detector_preserves_caller_row_order() -> None:
    train, test = synthetic_store(seed=1), synthetic_store(seed=2)
    detector = build("lstm_autoencoder")
    detector.params["epochs"] = 3
    scores = detector.fit_score(dataset_split(train, test), FEATURE_COLUMNS)

    assert len(scores) == len(test)
    assert not np.isnan(scores).any()
    assert np.all((scores >= 0) & (scores <= 1))


def test_deep_detectors_are_registered() -> None:
    assert {"autoencoder", "vae", "lstm_autoencoder"} <= set(available())


def test_sequence_detector_rejects_the_flat_interface() -> None:
    with pytest.raises(NotImplementedError, match="identity-ordered"):
        build("lstm_autoencoder").fit(pd.DataFrame({"a": [1.0]}))
