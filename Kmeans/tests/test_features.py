from pathlib import Path

from user_segmentation.config import load_config
from user_segmentation.data_loader import clean_tables, load_raw_tables
from user_segmentation.features import build_user_features
from scripts.generate_sample_data import generate_sample_data


def test_feature_builder_creates_expected_columns(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs" / "config.yaml")
    generate_sample_data(tmp_path, users=50)
    tables = clean_tables(load_raw_tables(config, data_dir=tmp_path))
    features = build_user_features(tables, config)

    for column in config.clustering_columns:
        assert column in features.columns

    assert len(features) == 50
