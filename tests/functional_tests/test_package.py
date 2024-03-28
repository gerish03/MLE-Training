def test_pkg_import():
    try:
        import HousePricePrediction  # noqa
    except Exception as e:
        assert False, (
            f"Error: {e}. "
            " HousePricePrediction package is not \
                imported and installed correctly."
        )


def test_import_ingest_data():
    try:
        from HousePricePrediction import ingest_data
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. ingest_data module is not imported correctly."
    try:
        assert ingest_data is not None
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is None."

    try:
        assert hasattr(ingest_data, "fetch_housing_data")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. ingest_data module doesn't have \
            'fetch_housing_data' attribute."

    try:
        assert hasattr(ingest_data, "load_housing_data")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. ingest_data module doesn't have \
            'load_housing_data' attribute."

    try:
        assert hasattr(ingest_data, "prepare_data_for_training")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. ingest_data module doesn't have \
            'prepare_data_for_training' attribute."


def test_import_score():
    try:
        from HousePricePrediction import score
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. score module is not \
            imported correctly."

    try:
        assert score is not None
    except Exception as e:
        assert False, f"Error: {e}. score is None."

    try:
        assert hasattr(score, "RF_score")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. score module doesn't have \
            'RF_score' attribute."

    try:
        assert hasattr(score, "score_model_mae")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. score module doesn't have \
            'score_model_mae' attribute."

    try:
        assert hasattr(score, "score_model_rmse")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. score module doesn't have \
            'score_model_rmse' attribute."


def test_import_train():
    try:
        from HousePricePrediction import train
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. train module is \
            not imported correctly."

    try:
        assert train is not None
    except Exception as e:
        assert False, f"Error: {e}. train is None."

    try:
        assert hasattr(train, "train_decision_tree")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. train module doesn't have \
            'train_decision_tree' attribute."

    try:
        assert hasattr(train, "train_linear_regression")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. train module doesn't have \
            'train_linear_regression' attribute."

    try:
        assert hasattr(train, "rand_tune_random_forest")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. train module doesn't have \
            'rand_tune_random_forest' attribute."

    try:
        assert hasattr(train, "grid_tune_random_forest")
    except Exception as e:
        assert (
            False
        ), f"Error: {e}. train module doesn't have \
            'grid_tune_random_forest' attribute."
