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
# pylint: disable=too-many-lines, redefined-builtin, too-many-locals

"""Create window features from config.

A function that takes a spark dataframe and creates window features based on a config
"""
import functools
import logging
from inspect import signature
from typing import Any, Callable, Dict, List, Optional, Union

import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Window

from feature_generation.v1.core.timeseries.array_aggregate import aggregate_over_slice

logger = logging.getLogger(__name__)


def generate_windows_spec(  # pylint: disable=too-many-branches  # noqa: C901
    partition_by: Union[List[str], str],
    order_by: Optional[Union[List[str], str]] = None,
    range_between: Optional[
        Union[List[Union[int, str]], List[List[Union[int, str]]]]
    ] = None,
    rows_between: Optional[
        Union[List[Union[int, str]], List[List[Union[int, str]]]]
    ] = None,
    descending: bool = False,
) -> List[pyspark.sql.WindowSpec]:
    """Creates a list of `WindowSpec` objects.

    Args:
        partition_by: Partitioning column for window
        order_by: Ordering column in the window
        range_between:  A tuple (list of 2 elements in yaml) for
        `Window.rangeBetween`'s ``range_start`` and ``range_end``. It can also be
        a list of tuples, in case you want to create multiple new columns for
        different ranges.
        rows_between: Similar to ``range_between``. A tuple (list of 2 elements in
        yaml) for `Window.rowsBetween`'s ``rows_start`` and ``rows_end``. It can
        also be a list of tuples, in case you want to create multiple new columns
        for different ranges.
        descending: Whether to reverse the order of the window.

    Note:
        "Unbounded Preceding" and "Unbounded Following" can be used as an element of
        ``range_between`` and ``rows_between`` to represent
        pyspark.sql.Window.unboundedPreceding and pyspark.sql.Window.unboundedFollowing
        respectively.

    Returns:
        A list of spark `WindowSpec` objects.
    """
    if isinstance(partition_by, str):
        partition_by = [partition_by]

    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        if descending:
            order_by = [f.col(x).desc() for x in order_by]

    ranges = range_between or rows_between
    if ranges:
        if not isinstance(ranges[0], list):
            ranges = [ranges]

        parsed_ranges = [
            _convert_range_definition(window_range) for window_range in ranges
        ]

    windows_spec = []
    if ranges:
        for window_range in parsed_ranges:
            window = Window.partitionBy(*partition_by)

            if order_by:
                window = window.orderBy(*order_by)

            if range_between:
                window = window.rangeBetween(window_range[0], window_range[1])
            else:
                window = window.rowsBetween(window_range[0], window_range[1])
            windows_spec.append(window)
    else:
        window = Window.partitionBy(*partition_by)

        if order_by:
            window = window.orderBy(*order_by)

        windows_spec.append(window)
    return windows_spec


def window_column(  # noqa: C901
    outputs: Union[str, List[str]],
    input: pyspark.sql.Column,
    windows_spec: Union[pyspark.sql.WindowSpec, List[pyspark.sql.WindowSpec]],
) -> List[pyspark.sql.Column]:
    """Creates a new column given the window configuration.

    Args:
        outputs: Output column name, or list of column names if multiple tuples are
        provided for ``range_between``.
        input: Input column.
        windows_spec: PySpark `WindowSpec` objects.

    Note:
        "Unbounded Preceding" and "Unbounded Following" can be used as an element of
        ``range_between`` and ``rows_between`` to represent
        pyspark.sql.Window.unboundedPreceding and pyspark.sql.Window.unboundedFollowing
        respectively.

    Returns:
        A list of spark column objects.

    Raises:
        ValueError: When length of ``outputs`` does not match length of
        ``range_between``.
    """
    if isinstance(windows_spec, pyspark.sql.WindowSpec):
        windows_spec = [windows_spec]

    if isinstance(outputs, str):
        outputs = [outputs]

    if len(outputs) != len(windows_spec):
        raise ValueError(
            "If you have multiple windows for a column, ensure ``output`` and "
            f"``windows_spec`` have the same length. There are {len(outputs)} names "
            f"but {len(windows_spec)} windows_spec"
        )

    columns = []
    for output_col, window_spec in zip(outputs, windows_spec):
        new_column = input.over(window_spec).alias(output_col)
        columns.append(new_column)

    return columns


def _check_input(agg_function: Callable, input_param: str):
    """Check if an input can be called by callable."""
    sig = signature(agg_function)
    if len(sig.parameters) == 0:  # pylint: disable=no-else-return
        logger.info(
            """No input required for aggregate function.
            Multiple calls will generate repeated columns."""
        )
        return agg_function()
    else:
        return agg_function(input_param)


# pylint: disable=too-many-arguments
def generate_window_grid(
    funcs: List[Callable],
    windows: List[Dict[str, List[str]]],
    inputs: Optional[List[str]] = None,
    ranges_between: Optional[List[List[Union[int, str]]]] = None,
    rows_between: Optional[List[List[Union[int, str]]]] = None,
    positive_term: str = "next",
    negative_term: str = "past",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> Union[List[Dict[Any, Any]], List[pyspark.sql.Column]]:
    """Generates window columns given a grid.

    Grid should contain columns, aggregation functions, windows, and ranges. Note that
    the user must ensure compatibility of all combinations within the grid.

    Args:
        inputs: A list of input columns. E.g.: ``["x_flag", "y_flag"]``.
        funcs: A list of input aggregation functions from
        ``pyspark.sql.functions``. E.g.: ``["min", "max", "sum"]``.
        windows: A list of dictionaries of window definition. E.g.:
        [
        {"partition_by": ["name"], "order_by": ["date_index"]},
        {"partition_by": ["name"], "order_by": ["date_index"]},
        ]
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        rows_between: A list of list of window ranges for ``rows_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when
        column names may seem odd! Defaulting to "between".
        positive_default: A string which would replace positive_term when
        column names may seem odd! Defaulting to "and".
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        A list of columns that can be passed to ``create_columns_from_config``.

    Raises:
        ValueError: If both or none of ``range_between`` or ``rows_between`` are
            provided.
    """
    ranges = ranges_between or rows_between
    inputs = inputs or [""]

    if not ranges:
        raise ValueError("Please supply either ``ranges_between`` or ``rows_between``.")

    expanded_list = []
    for _input in inputs:
        for _window in windows:
            for _agg_function in funcs:
                for _range in ranges:
                    (
                        output_col_name,
                        windows_spec,
                    ) = _create_window_attributes(
                        ranges=_range,
                        input_column=_input,
                        agg_function=_agg_function,
                        prefix=prefix,
                        suffix=suffix,
                        negative_term=negative_term,
                        positive_term=positive_term,
                        negative_default=negative_default,
                        positive_default=positive_default,
                        ranges_between=ranges_between,
                        window=_window,
                    )

                    expanded_list.extend(
                        window_column(
                            outputs=output_col_name,
                            input=_check_input(_agg_function, _input),
                            windows_spec=windows_spec,
                        )
                    )

    return expanded_list


def _generate_window_column_name(
    input_col_name: str,
    agg_function_name: str,
    prefix: str,
    suffix: str,
    window_range: List[Union[int, pyspark.sql.Window]],
    negative_term: str,
    positive_term: str,
    negative_default: str,
    positive_default: str,
) -> str:
    """Given a window range, attempts to produce a sensible window column name.

    Args:
        input_col_name: The column name.
        agg_function_name: The aggregation function name to apply within the window.
        prefix: Any prefix to be added to the output column names.
        suffix: Any suffix to be added to the output column names.
        window_range: A list representing the window range. E.g.:
        [-1, 1], [-1. Window.unboundedFollowing], [7, 14] etc.
        negative_term: The string to use if there is a negative window range.
        positive_term: The string to use if there is a positive window range.

    Returns:
        A string representing the column name.
    """
    column_range_substring = _generate_window_column_range_substring(
        suffix=suffix,
        window_range=window_range,
        negative_term=negative_term,
        positive_term=positive_term,
        negative_default=negative_default,
        positive_default=positive_default,
    )
    # pylint: enable=invalid-name
    # pylint: disable=line-too-long
    output_col_name = f"{prefix}{input_col_name}_{agg_function_name}_{column_range_substring}"  # noqa: E501
    return output_col_name
    # pylint: enable=line-too-long


def _convert_range_definition(
    window_range: List[Union[int, str]]
) -> List[Union[int, pyspark.sql.Window]]:
    """Convert window range definition.

    The element of ``window_range`` will be converted as below.
    - "UNBOUNDED PRECEDING": Window.unboundedPreceding
    - "UNBOUNDED FOLLOWING": Window.unboundedFollowing
    - Other than above: original value

    Args:
        window_range: The list representing window range

    Returns:
        The converted window range
    """
    range_dict = {
        "UNBOUNDED PRECEDING": Window.unboundedPreceding,
        "UNBOUNDED FOLLOWING": Window.unboundedFollowing,
    }
    return [int(range_dict.get(str(e).upper(), e)) for e in window_range]


def _generate_window_column_range_substring(
    suffix: str,
    window_range: List[Union[int, pyspark.sql.Window]],
    negative_term: str,
    positive_term: str,
    negative_default: str,
    positive_default: str,
) -> str:
    """Given a window range, attempts to produce a sensible window range suffix name.

    Args:
        suffix: Any suffix to be added to the output column names.
        window_range: A list representing the window range. E.g.:
        [-1, 1], [-1. Window.unboundedFollowing], [7, 14] etc.
        negative_term: The string to use if there is a negative window range.
        positive_term: The string to use if there is a positive window range.

    Returns:
        A string representing the range suffix for column name.
    """
    unbounded_preceding = "unbounded_preceding"
    unbounded_following = "unbounded_following_"
    if window_range[0] < 0:  # noqa: SIM114
        term_1 = negative_term
    elif window_range[1] > window_range[0] == 0:
        term_1 = negative_term
    elif window_range[0] > 0:
        term_1 = positive_term
    else:
        logger.info(
            f"Defaulting term_1 to {negative_default}, column names may seem odd!"
        )
        term_1 = negative_default

    if window_range[1] < 0:
        term_2 = negative_term
    elif window_range[0] < window_range[1] == 0:  # noqa: SIM114
        term_2 = positive_term
    elif window_range[1] > 0:
        term_2 = positive_term
    else:
        logger.info(
            f"Defaulting term_2 to {positive_default}, column names may seem odd!"
        )
        term_2 = positive_default

    range1 = (
        unbounded_preceding
        if window_range[0] == Window.unboundedPreceding
        else str(abs(window_range[0]))
    )
    range2 = (
        unbounded_following
        if window_range[1] == Window.unboundedFollowing
        else str(abs(window_range[1]))
    )
    # pylint: enable=invalid-name
    output_range_substring = (
        f"{term_1}_{range1}_{term_2}_{range2}{suffix}"  # noqa: E501
    )
    return output_range_substring


def generate_distinct_element_window_grid(  # pylint: disable=invalid-name
    inputs: List[str],
    windows: List[Dict[str, List[str]]],
    ranges_between: Optional[List[List[Union[int, str]]]] = None,
    rows_between: Optional[List[List[Union[int, str]]]] = None,
    positive_term: str = "next",
    negative_term: str = "past",
    agg_func_name: str = "count",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> Union[List[Dict[Any, Any]], List[pyspark.sql.Column]]:
    """Generates window columns with distinct elements in given a grid.

    Grid should contain columns, windows, and ranges. Note that
    the input columnn should only have array of elements we want
    to calculate over a grid.

    Args:
        inputs: A list of input columns. Each column must be an array
        containing the distinct elements to count.
        windows: A list of dictionaries of window definition. E.g.:
        [
        {"partition_by": ["name"], "order_by": ["date_index"]},
        {"partition_by": ["name"], "order_by": ["date_index"]},
        ]
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        rows_between: A list of list of window ranges for ``rows_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when
        column names may seem odd! Defaulting to "between".
        positive_default: A string which would replace positive_term when
        column names may seem odd! Defaulting to "and".
        agg_func_name: A string name for aggregate function
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        A list of columns that can be passed to ``create_columns_from_config``.

    Raises:
        ValueError: If both or none of ``range_between`` or ``rows_between`` are
        provided.
    """
    ranges = ranges_between or rows_between

    if not ranges:
        raise ValueError("Please supply either ``ranges_between`` or ``rows_between``.")

    def _dummy_func(x):
        return x

    _dummy_func.__name__ = agg_func_name

    expanded_list = []
    for _input in inputs:
        for _window in windows:
            for _range in ranges:
                (
                    output_col_name,
                    windows_spec,
                ) = _create_window_attributes(
                    ranges=_range,
                    input_column=_input,
                    agg_function=_dummy_func,
                    prefix=prefix,
                    suffix=suffix,
                    negative_term=negative_term,
                    positive_term=positive_term,
                    negative_default=negative_default,
                    positive_default=positive_default,
                    ranges_between=ranges_between,
                    window=_window,
                )

                expanded_list.extend(
                    size_collect_set(
                        outputs=output_col_name,
                        input_column=_input,
                        windows_spec=windows_spec,
                    )
                )

    return expanded_list


def size_collect_set(
    input_column: pyspark.sql.Column,
    outputs: Union[str, List[str]],
    windows_spec: Union[pyspark.sql.WindowSpec, List[pyspark.sql.WindowSpec]],
) -> List[pyspark.sql.Column]:
    """Function string to compute distinct elements' count over a partition in a window.

    Args:
         input_column: input column for aggregation.
         outputs: output column name.
         windows_spec: list of spark `WindowSpec` object.

    Returns:
         Spark column object to compute count of distinct list elements over windows.

    Raises:
        ValueError: When length of ``outputs`` does not match length of
        ``range_between``.
    """
    if isinstance(windows_spec, pyspark.sql.WindowSpec):
        windows_spec = [windows_spec]

    if isinstance(outputs, str):
        outputs = [outputs]

    if len(outputs) != len(windows_spec):
        raise ValueError(
            "If you have multiple windows for a column, ensure ``output`` and "
            f"``windows_spec`` have the same length. There are {len(outputs)} names "
            f"but {len(windows_spec)} windows_spec"
        )

    columns = []
    for output_col, window_spec in zip(outputs, windows_spec):
        new_column = f.size(
            f.array_distinct(f.flatten(f.collect_list(input_column).over(window_spec)))
        ).alias(output_col)
        columns.append(new_column)

    return columns


def _create_window_attributes(
    ranges,
    input_column,
    agg_function,
    prefix,
    suffix,
    negative_term,
    positive_term,
    negative_default,
    positive_default,
    ranges_between,
    window,
):
    """Generates window related attributes - Output column name and window specs.

    Args:
        ranges: Window range.
        input_column: The column name.
        agg_function: The aggregation function name to apply within the window.
        prefix: Any prefix to be added to the output column names.
        suffix: Any suffix to be added to the output column names.
        negative_term: The string to use if there is a negative window range.
        positive_term: The string to use if there is a positive window range.
        negative_default: A string which would replace negative_term when
        column names may seem odd.
        positive_default: A string which would replace positive_term when
        column names may seem odd.
        ranges_between: Window range for ``range_between``.
        window: Dictionary of window definition

    Returns:
        Output column name and window specs.
    """
    window_range = _convert_range_definition(ranges)

    if isinstance(agg_function, functools.partial):
        agg_function_name = agg_function.func.__name__
    else:
        agg_function_name = agg_function.__name__

    output_col_name = _generate_window_column_name(
        input_col_name=input_column,
        agg_function_name=agg_function_name,
        prefix=prefix,
        suffix=suffix,
        window_range=window_range,
        negative_term=negative_term,
        positive_term=positive_term,
        negative_default=negative_default,
        positive_default=positive_default,
    )

    range_arg = "range_between" if ranges_between else "rows_between"

    windows_spec = generate_windows_spec(
        **{
            "partition_by": window.get("partition_by"),
            "order_by": window.get("order_by"),
            range_arg: ranges,
            "descending": window.get("descending", False),
        }
    )

    return output_col_name, windows_spec


# pylint: disable=too-many-locals
def generate_window_delta(
    inputs: List[str],
    funcs: List[Callable],
    windows: List[Dict[str, List[str]]],
    ranges_between: Optional[List[List[Union[int, str]]]] = None,
    rows_between: Optional[List[List[Union[int, str]]]] = None,
    positive_term: str = "next",
    negative_term: str = "past",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> List[pyspark.sql.Column]:
    """Generates delta of window columns given a grid.

    Grid should contain columns, aggregation functions, windows, and ranges. Note that
    the user must ensure compatibility of all combinations within the grid.

    Args:
        inputs: A list of input columns. E.g.: ``["x_flag", "y_flag"]``.
        funcs: A list of input aggregation functions from
        ``pyspark.sql.functions``. E.g.: ``["min", "max", "sum"]``.
        windows: A list of dictionaries of window definition. E.g.:
        [
        {"partition_by": ["name"], "order_by": ["date_index"]},
        {"partition_by": ["name"], "order_by": ["date_index"]},
        ]
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        rows_between: A list of list of window ranges for ``rows_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when
        column names may seem odd. Defaulting to "between".
        positive_default: A string which would replace positive_term when
        column names may seem odd. Defaulting to "and".
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        A list of window columns and the delta between the two consecutive ranges.
        So, if you provide [wr1, wr2, wr3, wr4], the function will do delta for
        [wr2 - wr1, wr3 - wr2, wr4 - wr3] and return [wr1, wr2, wr3, wr4] as well.

    Raises:
        ValueError: If less than 2 ranges are provided
    """
    ranges = ranges_between or rows_between

    if len(ranges) < 2:
        raise ValueError("At least 2 window ranges should be provided.")

    delta_output = []

    # Generate window column name components
    for _input in inputs:
        for _window in windows:
            for _agg_function in funcs:
                if isinstance(_agg_function, functools.partial):
                    agg_function_name = _agg_function.func.__name__
                else:
                    agg_function_name = _agg_function.__name__
                initial_col_name = f"{prefix}delta_{_input}_{agg_function_name}"
                column_components_list = []
                for _range in ranges:
                    (
                        output_col_name,
                        windows_spec,
                    ) = _create_window_attributes(
                        ranges=_range,
                        input_column=_input,
                        agg_function=_agg_function,
                        prefix=prefix,
                        suffix=suffix,
                        negative_term=negative_term,
                        positive_term=positive_term,
                        negative_default=negative_default,
                        positive_default=positive_default,
                        ranges_between=ranges_between,
                        window=_window,
                    )

                    window_range = _convert_range_definition(_range)
                    output_range_substr = _generate_window_column_range_substring(
                        suffix=suffix,
                        window_range=window_range,
                        negative_term=negative_term,
                        positive_term=positive_term,
                        negative_default=negative_default,
                        positive_default=positive_default,
                    )
                    column_components_list.append(
                        {
                            "col": window_column(
                                outputs=output_col_name,
                                input=_agg_function(_input),
                                windows_spec=windows_spec,
                            )[
                                0
                            ],  # since window_column returns a list
                            "initial_col_name": initial_col_name,
                            "output_range_substr": output_range_substr,
                        }
                    )

                # generate delta columns
                delta_output.extend(
                    [
                        column_components["col"]
                        for column_components in column_components_list
                    ]
                )
                delta_output.extend(
                    (
                        column_components_list[i + 1]["col"]
                        - column_components_list[i]["col"]
                    ).alias(
                        f"{column_components_list[i + 1]['initial_col_name']}_"
                        f"{column_components_list[i + 1]['output_range_substr']}"
                        f"_and_{column_components_list[i]['output_range_substr']}"
                    )
                    for i in range(len(column_components_list) - 1)
                )

    return delta_output


def generate_window_ratio(
    inputs: Dict[Any, Any],
    funcs: List[Callable],
    windows: List[Dict[str, List[str]]],
    ranges_between: Optional[List[List[Union[int, str]]]] = None,
    rows_between: Optional[List[List[Union[int, str]]]] = None,
    positive_term: str = "next",
    negative_term: str = "past",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> List[pyspark.sql.Column]:
    """Generates ratio of window columns given a grid.

    Grid should contain columns, aggregation functions, windows, and ranges. Note that
    the user must ensure compatibility of all combinations within the grid.
    Args:
        inputs: A dictionary of input columns.
        E.g.: ``{'numerator_col1': 'denominator_column1',
        'numerator_col2': 'denominator_column2'}``.
        funcs: A list of input aggregation functions from
        ``pyspark.sql.functions``. E.g.: ``["min", "max", "sum"]``.
        windows: A list of dictionaries of window definition. E.g.:
        [
        {"partition_by": ["name"], "order_by": ["date_index"]},
        {"partition_by": ["name"], "order_by": ["date_index"]},
        ]
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        rows_between: A list of list of window ranges for ``rows_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when
        column names may seem odd. Defaulting to "between".
        positive_default: A string which would replace positive_term when
        column names may seem odd. Defaulting to "and".
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        Returns a list of window columns and the ratio between the window columns.``.
            So, if you provide input ``{'numerator_col1': 'denominator_column1'}``
            it wil do ratio for numerator_col1_window/denominator_column1_window

    Raises:
        ValueError: If both or none of ``range_between`` or ``rows_between`` are
        provided.
        TypeError: If dict is not provided as input.
    """
    ranges = ranges_between or rows_between

    if not ranges:
        raise ValueError("Please supply either ``ranges_between`` or ``rows_between``.")

    if not isinstance(inputs, dict):
        raise TypeError("""Argument ``inputs`` should be of dict type.""")

    ratio_list = []
    # creating window_cols as dict to avoid duplication of window columns objects
    # for cases where denominator of multiple mappings is same
    window_cols = {}
    for numerator_denominator_pair in inputs.items():
        for _window in windows:
            for _agg_function in funcs:
                for _range in ranges:
                    ratio_input_cols = []
                    for column_name in numerator_denominator_pair:
                        (
                            output_col_name,
                            windows_spec,
                        ) = _create_window_attributes(
                            ranges=_range,
                            input_column=column_name,
                            agg_function=_agg_function,
                            prefix=prefix,
                            suffix=suffix,
                            negative_term=negative_term,
                            positive_term=positive_term,
                            negative_default=negative_default,
                            positive_default=positive_default,
                            ranges_between=ranges_between,
                            window=_window,
                        )

                        # denominator columns will get computed once
                        window_cols[output_col_name] = window_column(
                            outputs=output_col_name,
                            input=_agg_function(column_name),
                            windows_spec=windows_spec,
                        )
                        ratio_input_cols.append(
                            {
                                "col": window_cols[output_col_name][0],
                                "column_name": output_col_name.replace(prefix, ""),
                            }
                        )

                    ratio_list.append(
                        (ratio_input_cols[0]["col"] / ratio_input_cols[1]["col"]).alias(
                            f"{prefix}ratio_{ratio_input_cols[0]['column_name']}"
                            f"_{ratio_input_cols[1]['column_name']}"
                        )
                    )

    for column_obj in window_cols.values():
        ratio_list.extend(column_obj)

    return ratio_list


# pylint: disable=too-many-arguments,unexpected-keyword-arg
def aggregate_over_slice_grid(
    inputs: List[str],
    anchor_col: str,
    anchor_array: List[str],
    funcs: List[str],
    ranges_between: List[List[Union[int, str]]],
    positive_term: str = "next",
    negative_term: str = "past",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> List[pyspark.sql.Column]:
    """Generates slices (window column equivalent) given a grid.

    Grid should contain columns, aggregation functions and ranges. Note that
    the user must ensure compatibility of all combinations within the grid.

    Args:
        inputs: A list of input columns. E.g.: ``["x_flag", "y_flag"]``.
        anchor_col: Anchor column.
        anchor_array: Array column with all the anchor column values.
        funcs: A list of input aggregation functions that work on arrays.
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when column
        names may seem odd! Defaulting to "between".
        positive_default: A string which would replace positive_term when column
        names may seem odd! Defaulting to "and".
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        A list of columns.
    """
    expanded_list = []
    for _input in inputs:
        for _agg_function in funcs:
            if isinstance(_agg_function, functools.partial):
                agg_function_name = _agg_function.func.__name__
            else:
                agg_function_name = _agg_function.__name__
            for _range in ranges_between:
                output_col_name = _generate_window_column_name(
                    input_col_name=_input,
                    agg_function_name=agg_function_name,
                    prefix=prefix,
                    suffix=suffix,
                    window_range=_range,
                    negative_term=negative_term,
                    positive_term=positive_term,
                    negative_default=negative_default,
                    positive_default=positive_default,
                )

                expanded_list.append(
                    aggregate_over_slice(
                        input_col=_input,
                        lower_bound=min(_range),
                        upper_bound=max(_range),
                        anchor=anchor_col,
                        anchor_array=anchor_array,
                        func=_agg_function,
                        alias=output_col_name,
                    )
                )

    return expanded_list


def generate_array_elements_window_grid(  # pylint: disable=invalid-name
    inputs: List[str],
    windows: List[Dict[str, List[str]]],
    agg_functions: List[Callable],
    ranges_between: Optional[List[List[Union[int, str]]]] = None,
    rows_between: Optional[List[List[Union[int, str]]]] = None,
    positive_term: str = "next",
    negative_term: str = "past",
    negative_default: str = "between",
    positive_default: str = "and",
    prefix: str = "",
    suffix: str = "",
) -> Union[List[Dict[Any, Any]], List[pyspark.sql.Column]]:
    """Generates window columns with distinct array elements in given a grid.

    Grid should contain columns, windows, and ranges. Note that
    the input columnn should only have array of elements we want
    to calculate over a grid.

    Args:
        inputs: A list of input columns. Each column must be an array
        containing the distinct elements to count.
        windows: A list of dictionaries of window definition. E.g.:
        [
        {"partition_by": ["name"], "order_by": ["date_index"]},
        {"partition_by": ["name"], "order_by": ["date_index"]},
        ]
        ranges_between: A list of list of window ranges for ``range_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        rows_between: A list of list of window ranges for ``rows_between``.
        E.g.: [[-1, -1], [0, 0], [1, 1]]. The following strings are also accepted:
        "UNBOUNDED PRECEDING" and "UNBOUNDED FOLLOWING". E.g.
        [["UNBOUNDED PRECEDING", 6], [-4, "UNBOUNDED FOLLOWING"]].
        positive_term: A string name for the positive term when constructing a
        new column name. Defaults to "next".
        negative_term: A string name for the negative term when constructing a
        new column name. Defaults to "past".
        negative_default: A string which would replace negative_term when column
        names may seem odd! Defaulting to "between".
        positive_default: A string which would replace positive_term when column
        names may seem odd! Defaulting to "and".
        agg_functions: List of Callable aggregation function.
        prefix: A string name for the prefix when constructing a new column name.
        Defaults to empty string "".
        suffix: A string name for the suffix when constructing a new column name.
        Defaults to empty string "".

    Returns:
        A list of columns that can be passed to ``create_columns_from_config``.

    Raises:
        ValueError: If both or none of ``range_between`` or ``rows_between`` are
        provided.
    """
    ranges = ranges_between or rows_between

    if not ranges or (ranges_between and rows_between):
        raise ValueError("Please supply either ``ranges_between`` or ``rows_between``.")

    expanded_list = []
    for _input in inputs:
        for _window in windows:
            for agg_function in agg_functions:
                for _range in ranges:
                    (
                        output_col_name,
                        windows_spec,
                    ) = _create_window_attributes(
                        ranges=_range,
                        input_column=_input,
                        agg_function=agg_function,
                        prefix=prefix,
                        suffix=suffix,
                        negative_term=negative_term,
                        positive_term=positive_term,
                        negative_default=negative_default,
                        positive_default=positive_default,
                        ranges_between=ranges_between,
                        window=_window,
                    )

                    expanded_list.extend(
                        aggregate_over_collect_list(
                            outputs=output_col_name,
                            input_column=_input,
                            windows_spec=windows_spec,
                            agg_function=agg_function,
                        )
                    )

    return expanded_list


def aggregate_over_collect_list(
    input_column: pyspark.sql.Column,
    outputs: Union[str, List[str]],
    windows_spec: Union[pyspark.sql.WindowSpec, List[pyspark.sql.WindowSpec]],
    agg_function: Callable,
) -> List[pyspark.sql.Column]:
    """Aggregate distinct array elements (col_name, window_spec) attributes over window.

    Args:
         input_column: input column for aggregation.
         outputs: output column name.
         windows_spec: list of spark `WindowSpec` object.
         agg_function: Aggregation to be performed
    Returns:
         Spark column object to compute count of distinct list elements over windows.

    Raises:
        ValueError: When length of ``outputs`` does not match length of
        ``range_between``.
    """
    if isinstance(windows_spec, pyspark.sql.WindowSpec):
        windows_spec = [windows_spec]

    if isinstance(outputs, str):
        outputs = [outputs]

    if len(outputs) != len(windows_spec):
        raise ValueError(
            "If you have multiple windows for a column, ensure ``output`` and "
            f"``windows_spec`` have the same length. There are {len(outputs)} names "
            f"but {len(windows_spec)} windows_spec"
        )

    columns = []
    for output_col, window_spec in zip(outputs, windows_spec):
        new_column = agg_function(
            f.flatten(f.collect_list(input_column).over(window_spec))
        ).alias(output_col)
        columns.append(new_column)
    return columns
