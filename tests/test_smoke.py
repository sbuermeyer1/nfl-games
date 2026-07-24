from nfl_game import paths


def test_paths_resolve_under_project_root():
    assert paths.RAW_DIR == paths.PROJECT_ROOT / "data" / "raw"
    assert paths.PROCESSED_DIR == paths.PROJECT_ROOT / "data" / "processed"


def test_data_dirs_exist():
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DIR.is_dir()
