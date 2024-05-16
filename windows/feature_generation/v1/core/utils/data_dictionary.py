# (c) McKinsey & Company 2016 – Present
# All rights reserved
#
#
# This material is intended solely for your internal use and may not be reproduced,
# disclosed or distributed without McKinsey & Company's express prior written consent.
# Except as otherwise stated, the Deliverables are provided ‘as is’, without any express
# or implied warranty, and McKinsey shall not be obligated to maintain, support, host,
# update, or correct the Deliverables. Client guarantees that McKinsey’s use of
# information provided by Client as authorised herein will not violate any law
# or contractual right of a third party. Client is responsible for the operation
# and security of its operating environment. Client is responsible for performing final
# testing (including security testing and assessment) of the code, model validation,
# and final implementation of any model in a production environment. McKinsey is not
# liable for modifications made to Deliverables by anyone other than McKinsey
# personnel, (ii) for use of any Deliverables in a live production environment or
# (iii) for use of the Deliverables by third parties; or
# (iv) the use of the Deliverables for a purpose other than the intended use
# case covered by the agreement with the Client.
# Client warrants that it will not use the Deliverables in a "closed-loop" system,
# including where no Client employee or agent is materially involved in implementing
# the Deliverables and/or insights derived from the Deliverables.
# pylint: disable=too-many-locals
"""Creates data dictionary for a spark dataframe."""

import logging
import re
from typing import List, Mapping

import pyspark
from pyspark.sql import functions as f

from ...core.utils.strings import mask_string  # noqa: E501

# pylint: enable=line-too-long

logger = logging.getLogger(__name__)


def _rows(
    df: pyspark.sql.DataFrame,
    sample_mode: str,
    random_limit: int = 1000,
    random_fraction: float = 0.2,
):
    def _take(df: pyspark.sql.DataFrame, name: str):
        res_col = df.limit(1).collect()
        res = res_col[0][name] if len(res_col) > 0 else None
        res = str(res) if res is not None else ""
        res = re.sub(r"[\n\t\r]", "", res[:500])
        return res

    if "random" in sample_mode:
        df = df.limit(random_limit)
    for name, dtype in df.dtypes:
        if sample_mode == "none":
            yield [name, dtype]
        elif sample_mode == "first":
            yield [name, dtype, _take(df, name)]
        elif sample_mode == "first_nnull":
            yield [
                name,
                dtype,
                _take(
                    df.filter(f.col(name).isNotNull()).filter(
                        f.col(name).cast("string") != ""
                    ),
                    name,
                ),
            ]
        elif sample_mode == "first_random":
            yield [name, dtype, _take(df.sample(fraction=random_fraction), name)]
        elif sample_mode == "first_random_nnull":
            yield [
                name,
                dtype,
                _take(
                    df.orderBy(f.rand())
                    .filter(f.col(name).isNotNull())
                    .filter(f.col(name).cast("string") != ""),
                    name,
                ),
            ]


# pylint: disable=line-too-long
def generate_md_dictionary_spark(
    df: pyspark.sql.DataFrame, instructions: Mapping[str, str]
) -> str:
    """Generates a data dictionary for a spark dataframe.

    This function generates a string writen in a markdown format. It expects a set of instructions as per the example bellow.

    ```
    name: "trials"                              # Title
    description: "Trial information."           # Description
    sample_mode: "first"                        # Sample mode to apply. Can be `none`, `first`, `first_nnull`, `first_random`, `first_random_nnull`. Default is `none`
    random_limit: 100                           # Limit to apply when selected `first_random` or `first_random_nnull`. default is 1000
    random_fraction: 0.5                        # Random fraction to apply when `first_random` or `first_random_nnull`. default is 0.2
    mask_columns: ["trials_id"]                 # Columns to mask. It doesn't mask any column iif not passed
    ```

    The final table will have the format:

        +-------+-----+--------+
        |Column |Type | Sample |
        +-------+-----+--------+
        | xxxxx | xxx | xxxxxx |
        +-------+-----+--------+

    Args:
        df: spark dataframe to create the data dictionary.
        instructions: instructing about the dataframe, such as name, description and  sampling params.

    Returns:
        String in markdown style with a data dictionary.

    Raises:
        Exception: If selected a not supported sample_mode.
    """  # noqa: E501
    # pylint: enable=line-too-long
    supported_modes = [
        "none",
        "first",
        "first_nnull",
        "first_random",
        "first_random_nnull",
    ]
    name = instructions.get("name")
    description = instructions.get("description")
    sample_mode = instructions.get("sample_mode", "none")
    mask_columns = instructions.get("mask_columns", None)
    random_limit = instructions.get("random_limit", 1000)
    random_fraction = instructions.get("random_fraction", 0.2)

    if mask_columns:
        for column in mask_columns:
            df = df.withColumn(column, mask_string(column))

    if sample_mode not in supported_modes:
        raise Exception(  # pylint: disable=broad-exception-raised
            f"Mode: {sample_mode} is not supported. Supported: {supported_modes}"
        )
    top_header = f"# {name}\n {description}\n\n"
    row1 = ["Column", "Type"]
    row2 = ["---", "---"]

    if sample_mode != "none":
        row1.append("Sample")
        row2.append("---")
    rows = list(_rows(df, sample_mode, random_limit, random_fraction))
    all_rows = [row1, row2, *rows]

    tbl_txt = "\n".join([f"| {' | '.join(r)} |" for r in all_rows])
    txt = f"{top_header}{tbl_txt}"

    return txt


def generate_instructions(
    layer: str,
    datasets: List[str],
    metadata: Mapping[str, str],
    sample_mode: str = "first_nnull",
) -> Mapping[str, str]:
    """Generates instructions from metadata.

    Args:
        layer: data layer.
        datasets: list of catalog datasets.
        metadata: metadata about the datasets.
        sample_mode: sampling mode.

    Returns:
        Dataset instructions.

    Raises:
        Exception: If description is not provided for a dataset.
    """

    def _get_name(layer, dataset):
        """Outputs the dataset name."""
        if layer == "raw":
            [dataset, extract_type] = dataset.split("@")
            table_name = dataset.split(".")[-1]

            final_name = (
                f"{extract_type.title()} {layer} " f"{table_name[4:].replace('_', ' ')}"
            )

        else:
            layer_dict = {
                "raw": "Raw",
                "int": "Intermediate",
                "prm": "Primary",
            }

            table_name = dataset.split(".")[-1]
            final_name = f"{layer_dict[layer]} {table_name[4:].replace('_', ' ')}"

        return final_name

    data_docs_instructions = {}

    for dataset in datasets:
        dataset_metadata = metadata.get(dataset, {})

        if "description" not in dataset_metadata and layer not in ["raw", "int"]:
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Description should be provided for {dataset}"
            )

        data_docs_instructions[dataset] = {}
        data_docs_instructions[dataset]["name"] = dataset_metadata.get(
            "name", _get_name(layer, dataset)
        )

        data_docs_instructions[dataset]["description"] = dataset_metadata.get(
            "description", ""
        )
        data_docs_instructions[dataset]["sample_mode"] = dataset_metadata.get(
            "sample_mode", sample_mode
        )
        data_docs_instructions[dataset]["mask_column"] = dataset_metadata.get(
            "mask_columns", None
        )
    return data_docs_instructions


def generate_table_datasets(layer: str, instructions: Mapping[str, str]) -> str:
    """Generates a list of available tables.

    Args:
        layer: data layer.
        instructions: dataset instructions.

    Returns:
        String in markdown style with a list of datasets.
    """
    layer_dict = {
        "raw": "Raw layer",
        "int": "Intermediate layer",
        "prm": "Primary layer",
    }

    title = f"### {layer_dict[layer]}"

    header = """
| Dataset | Name |
| ------- | ---- |"""

    rows = ""
    for dataset_name, dataset_params in instructions.items():
        if layer != "raw" or "@load" in dataset_name:
            row = (
                f"| [{dataset_name}](./docs/dictionaries/{dataset_name}.md) "
                f"| {dataset_params['name']}"
            )
        else:
            row = f"| {dataset_name} | {dataset_params['name']}"

        rows = rows + "\n" + row

    return title + "\n" + header + rows
