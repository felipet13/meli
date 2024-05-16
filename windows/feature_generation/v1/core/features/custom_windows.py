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

"""Custom window aggregation functions."""
from typing import Callable, Optional

from pyspark.sql.functions import PandasUDFType, pandas_udf


def _perform_complete_op(
    n_rows: int,
    func: Callable,
    func_name: str,
    null_value: Optional[str] = None,
    return_dtype: str = "double",
):
    """Performs ``func`` operation only if entire range is complete.

    Returns a callable pyspark aggregation function that outputs values
    if the number of values is ``n_rows``, otherwise returns None/null.

    Args:
        n_rows: the number of rows to validate in each subset.
        func: python function aggregation over pd.Series.
        func_name: function name.
        null_value: value when entire range is not complete.
        return_dtype: dtype of pyspark column output.

    Returns:
        Callable: Callable function.
    """

    @pandas_udf(return_dtype, PandasUDFType.GROUPED_AGG)
    def udf_wrapper(v):
        if len(v) == n_rows:
            return func(v)
        else:
            return null_value

    udf_wrapper.__name__ = func_name
    return udf_wrapper


def complete_sum(
    n_rows: int, null_value: Optional[str] = None, return_dtype: str = "double"
) -> Callable:
    """Sum only if entire range is complete.

    Returns a callable pyspark aggregation function that sums all values
    if the number of values is ``n_rows``, otherwise returns ``null_value``

    Args:
        n_rows: the number of rows to validate in each subset.
        null_value: value when entire range is not complete.
        return_dtype: dtype of pyspark column output.

    Returns:
        Callable: Callable function.
    """
    return _perform_complete_op(
        n_rows=n_rows,
        func=lambda v: v.sum(),
        func_name="complete_sum",
        null_value=null_value,
        return_dtype=return_dtype,
    )


def complete_max(
    n_rows: int, null_value: Optional[str] = None, return_dtype: str = "double"
) -> Callable:
    """Take the maximum only if entire range is complete.

    Returns a callable pyspark aggregation function that takes the maximum of all values
    if the number of values is ``n_rows``, otherwise returns ``null_value``

    Args:
        n_rows: the number of rows to validate in each subset.
        null_value: value when entire range is not complete.
        return_dtype: dtype of pyspark column output.

    Returns:
        Callable: Callable function.
    """
    return _perform_complete_op(
        n_rows=n_rows,
        func=lambda v: v.max(),
        func_name="complete_max",
        null_value=null_value,
        return_dtype=return_dtype,
    )
