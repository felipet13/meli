"""Get current spark processing information."""
import logging
from abc import ABC

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


class SparkInfo(ABC):
    """Aggregates useful methods to allow investigation on Spark internal processing."""

    @staticmethod
    def log(df: DataFrame):
        """Entry point to get information from a resulting feature method dataframe."""
        logger.debug(
            "Number of RDD partitions: %d", SparkInfo._get_number_of_partitions(df)
        )
        logger.debug(SparkInfo._get_complete_spark_plan_text(df))

    @staticmethod
    def _get_number_of_partitions(df: DataFrame):
        """Get existing number of partitions of the current dataframe (in memory)."""
        return df.rdd.getNumPartitions()

    @staticmethod
    def _get_complete_spark_plan_text(df: DataFrame):
        """Get Spark plan of a dataframe.

        Given that `DataFrame.explain()` function explicitely `.print()` the Spark plan
        in `sys.stdout`, this function get the explain content from internal functions.
        This function gets both "extended" and "formatted", as they contains
        complementary info:
            - Extended gives parsed, analyzed, optimized (if available) and
              detailed physical plan;
            - Formatted gives physical plan and separated codegen section;
        """
        # Get explain content and remove the existing physical plan from it
        # This is a double lightweight operation and it does not harm the performance
        extended_output_spark_info = df._sc._jvm.PythonSQLUtils.explainString(
            df._jdf.queryExecution(), "extended"
        )

        formatted_output_spark_info = df._sc._jvm.PythonSQLUtils.explainString(
            df._jdf.queryExecution(), "formatted"
        )

        # Join both outputs (parsed/analyzed/optimized from extended +
        # physical/codegen from formatted)
        return (
            f"Generated Spark plan:\n\n"
            f"## Extended:\n{extended_output_spark_info}\n\n"
            f"## Formatted:\n{formatted_output_spark_info}"
        )
