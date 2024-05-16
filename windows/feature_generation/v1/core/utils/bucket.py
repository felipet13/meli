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

"""Contains bucket related utility functions.

This utility helps the user to generate bucket values over a numerical column
based on the configuration.

Compared to the core functionality provided by spark ML bucketizer, this
utility provide some advantages over it -
1. Does not break the lineage as it is simply a column callable function.
2. Gives user the configurability to generate even-ranged buckets automatically
based on the configuration.


df.withColumn('bucket_val', create_buckets('age', bucket_params))
+------+---+                                           +------+---+----------+
|   name|age|              create_buckets               |   name|age|bucket_val|
+------+---+     -------------------------------->     +------+---+----------+
| Cersei| 45|           bucket_params                   | Cersei| 40|   bucket4|
|    Jon| 25|              bucket1  [1, 15]             |    Jon| 25|   bucket2|
|  Sansa| 20|              bucket2  [15, 25]            |  Sansa| 20|   bucket2|
|   Arya| 16|              bucket3  [25, 35]            |   Arya| 16|   bucket2|
|  Jaime| 45|              bucket4  [35, 45]            |  Jaime| 45|   bucket4|
| Tyrion| 70|              bucket5  [45, 60]            | Tyrion| 75|  outliers|
+------+---+                                           +------+---+----------+


df.withColumn('bucket_val', create_even_ranged_buckets('age', bucket_params))
+------+---+                                            +------+---+------------+
|   name|age|         create_even_ranged_buckets         |   name|age|  bucket_val|
+------+---+     -------------------------------->      +------+---+------------+
| Cersei| 45|              bucket_params                 | Cersei| 40|20.0_to_40.0|
|    Jon| 25|                nim_val  0                  |    Jon| 25|20.0_to_40.0|
|  Sansa| 20|                max_val  60                 |  Sansa| 20| 0.0_to_20.0|
|   Arya| 16|                num_buckets  3              |   Arya| 16| 0.0_to_20.0|
|  Jaime| 45|                                            |  Jaime| 45|40.0_to_60.0|
| Tyrion| 70|                                            | Tyrion| 75|    outliers|
+------+---+                                            +------+---+------------+
"""


from typing import Any, Dict, List

import pyspark
import pyspark.sql.functions as f

from .alias import alias


@alias()
def create_buckets(
    column: str,
    bucket_params: Dict[str, List[Any]],
    default_bucket: str = "outliers",
) -> pyspark.sql.Column:
    """Create bucket column based on config paramters.

    Use `alias` argument for setting the name of the output column.

    ::
    # example
    bucket_params -
    bucket1: [1, 3]
    bucket2: [4, 6]
    bucket3: [7, 9]
    bucket4: [10, 20]

    Args:
        column: Column to bucket.
        bucket_params: Bucket instructions.
        default_bucket: Default bucket name.

    Returns:
        A case statement to compute bucket values.
    """
    bucket_params = bucket_params.copy()
    first_key = next(iter(bucket_params.keys()))

    first_range = bucket_params.pop(first_key)
    case_statement = f.when(
        f.col(column).between(float(first_range[0]), float(first_range[1])),
        f.lit(first_key),
    )

    for _bucket_name, _range in bucket_params.items():
        case_statement = case_statement.when(
            f.col(column).between(float(_range[0]), float(_range[1])),
            f.lit(_bucket_name),
        )
    return case_statement.otherwise(f.lit(default_bucket))


@alias()
def create_even_ranged_buckets(
    column: str,
    bucket_params: Dict[str, Any],
    default_bucket: str = "outliers",
) -> pyspark.sql.Column:
    """Create even-ranged bucket column based on config parameters.

    Use `alias` argument for setting the name of the output column.

    ::
    # example
    input:
    bucket_params
    min_val: 1
    max_val: 14
    num_buckets: 4

    output:
    1_to_4.25: [1, 4.25]
    4.25_to_7.5: (4.25, 7.5]
    7.5_to_10.25: (7.5, 10.75]
    10.25_to_14: (10.75, 14]

    Args:
        column: column to create buckets on
        bucket_params: Bucket instructions.
        default_bucket: Default bucket name.

    Returns:
        A case statement to compute even ranged bucket values.
    """
    min_val = float(bucket_params["min_val"])
    max_val = float(bucket_params["max_val"])
    num_buckets = int(bucket_params["num_buckets"])

    step_size = (max_val - min_val) / num_buckets
    steps = create_splits(min_val, max_val, num_buckets)

    results = {}
    for i in range(len(steps) - 1):
        _key = str(steps[i]) + "_to_" + str(steps[i] + step_size)
        _step = [steps[i], steps[i] + step_size]
        results[_key] = _step
    return create_buckets(column, results, default_bucket)


def create_splits(min_val: float, max_val: float, num_buckets: float) -> List[float]:
    """Create splits for given range and number of buckets.

    ::
    # example
    input:
    min_val = 1, max_val = 14, num_buckets = 4
    output:
    [1, 4.25, 7.5, 10.75, 14]

    Args:
        min_val: minimum value.
        max_val: maximum value.
        num_buckets: number of buckets.

    Returns:
        A list with equally spaced values.
    """
    step_size = (max_val - min_val) / num_buckets
    splits = [min_val + step_size * i for i in range(num_buckets + 1)]
    return splits
