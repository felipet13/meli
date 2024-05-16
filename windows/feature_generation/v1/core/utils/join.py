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

"""Join methods rewritten as functions."""

from functools import reduce
from typing import List, Union

import pyspark


def reduce_join(
    list_of_dataframes: List[pyspark.sql.DataFrame],
    join_key: Union[List[str], str],
    how: str = "left",
) -> pyspark.sql.DataFrame:
    """Joins a list of dataframes into a single dataframe on the specified join_key.

    Args:
        list_of_dataframes: A list of dataframes
        join_key: The join key for all the dataframes in ``list_of_dataframes``
        how: Spark SQL join type.

    Returns:
        A spark dataframe.
    """
    return reduce(lambda x, y: x.join(y, on=join_key, how=how), list_of_dataframes)
