"""Logger Tests."""

import logging

from pyspark.sql import DataFrame

from data_integration.v0.nodes import ingestor


class TestLogSparkInfo:
    def test_log_spark_info_true(self, input_df_transformation: DataFrame, caplog):
        """Test logger operation when log_spark_info is set to True."""

        instructions = {
            "rename_columns": {
                "birth_dt": "date_of_birth",
            }
        }

        with caplog.at_level(logging.DEBUG, logger="data_integration"):
            _ = ingestor.transform_data(
                input_df_transformation,
                instructions,
                log_spark_info=True,
            )

        print(f"{caplog.text[:500]=}")
        assert (
            "Number of RDD partitions:" in caplog.text
        ), "Log output does not contain number of partitions, but it should."
        assert (
            "Generated Spark plan:" in caplog.text
        ), "Log output does not contain initial message, but it should."
        assert (
            "== Parsed Logical Plan ==" in caplog.text
        ), "Log output does not contain `.explain()` elements, but it should."
        assert (
            "== Physical Plan ==" in caplog.text
        ), "Log output does not contain `.explain()` elements, but it should."

    def test_log_spark_info_false(self, input_df_transformation: DataFrame, caplog):
        """Test logger operation when log_spark_info is set to True."""

        instructions = {
            "rename_columns": {
                "birth_dt": "date_of_birth",
            }
        }

        with caplog.at_level(logging.DEBUG, logger="data_integration"):
            _ = ingestor.transform_data(
                input_df_transformation,
                instructions,  # log_spark_info not set, False by default
            )

        print(f"{caplog.text[:500]=}")
        assert (
            "Number of RDD partitions:" not in caplog.text
        ), "Number of partitions should not be present."
        assert (
            "== Parsed Logical Plan ==" not in caplog.text
        ), "Spark plan should not be present."
        assert (
            "== Physical Plan ==" not in caplog.text
        ), "Spark plan should not be present."
