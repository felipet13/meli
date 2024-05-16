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

"""API node for data aggregation."""

import pyspark
from refit.v1.core.fill_nulls import fill_nulls
from refit.v1.core.has_schema import has_schema
from refit.v1.core.inject import inject_object
from refit.v1.core.input_kwarg_filter import add_input_kwarg_filter
from refit.v1.core.input_kwarg_select import add_input_kwarg_select
from refit.v1.core.output_filter import add_output_filter
from refit.v1.core.retry import retry
from refit.v1.core.unpack import unpack_params

from ...core.aggregation import pivot


@add_input_kwarg_filter()
@add_input_kwarg_select()
@has_schema()
@unpack_params()
@inject_object()
@retry()
@add_output_filter()
@fill_nulls()
def pivot_on_multiple_fields(*args, **kwargs) -> pyspark.sql.DataFrame:
    """See core function for more details.

    Add_input_kwarg_filter, add_input_kwarg_select, has_schema, unpack_params,
    inject_object, retry, add_output_filter, fill_nulls
    wrapper for `pivot_on_multiple_fields`.
    """
    return pivot.pivot_on_multiple_fields(*args, **kwargs)
