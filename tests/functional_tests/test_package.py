def test_pkg_import():
    try:
        import HousePricePrediction  # noqa
    except Exception as e:
        assert False, (
            f"Error: {e}. "
            " mypackage package is not \
                imported and installed correctly."
        )