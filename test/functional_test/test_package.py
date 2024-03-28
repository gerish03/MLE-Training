def test_pkg_import():
    try:
        import HousePricePrediction  # noqa
    except Exception as e:
        assert False, (
            f"Error: {e}. "
            " mypackage package is not \
                imported and installed correctly."
        )

import HousePricePrediction
from HousePricePrediction import ingest_data, score, train


def test_pkg_import():
    assert HousePricePrediction is not None
    assert ingest_data is not None
    assert hasattr(ingest_data, "fetch_housing_data")
    assert hasattr(ingest_data, "load_housing_data")
    assert hasattr(ingest_data, "prepare_data_for_training")
    assert score is not None
    assert hasattr(score, "score_model")
    assert train is not None
    assert hasattr(train, "train_decision_tree")
    assert hasattr(train, "train_linear_regression")
    assert hasattr(train, "train_random_forest")
    assert hasattr(train, "tune_random_forest")