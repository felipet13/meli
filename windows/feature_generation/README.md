# Feature Generation

Feature generation utility aims for creating new features, potentially for
using in statistical analysis. This utility contains functionality split into
submodules to help build different feature types.

Note that the utility is based on spark.

## Business Problem

In critical Advanced Analytics projects, user would like to
create features quickly and with little repetition of boiler
plate code. The `feature_generation` utility attempts to address most of the common
features used in a machine learning use case. However, there should be a way to
also add custom features and combine the two seemlessly.


## Feature Generation Components



    +----+------------------------------------------+---------------------------------------------------------------------------+
    |    | sub_module                               | description                                                               |
    |----+------------------------------------------+---------------------------------------------------------------------------|
    |  0 | feature_generation.v1.core.aggregation   | Dataframe aggregation sub-module.                                         |
    |  1 | feature_generation.v1.core.features      | Contains a set of functions to build common features.                     |
    |  2 | feature_generation.v1.core.impute        | Contains a set of functions to perform imputation.                        |
    |  3 | feature_generation.v1.core.interpolation | Contains a set of functions to perform interpolation between data points. |
    |  4 | feature_generation.v1.core.tags          | Contains the tagging framework.                                           |
    |  5 | feature_generation.v1.core.timeseries    | Contains a set of functions for time-series analysis.                     |
    |  6 | feature_generation.v1.core.utils         | Contains a set of functions for common data cleaning activities.          |
    +----+------------------------------------------+---------------------------------------------------------------------------+

## Features Abstraction

When generating features, it is useful to be able to classify features into
abstract types, as opposed to more domain related groups of features.

The following is a simple classification of the type of features that can be created.
There are 2 major feature types - `basic` features and `derived` features.
Derived features are composed of mainly `window` and `interacted` features. You can
find more information below.


![Types of features](images/feature_types.png)

### Basic Features
As the name describes these features are created by a simple manipulation on the
original data. These can be created from most built-in pyspark functions:
```
x > y
avg(x)
x == 'drug_a'
```

### Derived Features
These features are created by transforming any base and/or other derived features. The
types of derived features can range from a simple combination of features, window of
a base feature, or complex functions according to a business rule. However, it is
important to note that derived features are features that are based on other features,
and not directly on the raw data.

There are 2 common groups of derived features: namely window features and interacted
features.

Note that the complex function according to business rule should belong to the scope of
the individual project being conducted.


#### Window features
These features are created in a way where the transformations are based on a windows.
For example:

* sum of X in the last Y days
* sum of drug flag in last 10 days (proxy for whether a drug was taken in the last 10 days)
* sum of sundays flags in next 30 days (proxy for a future looking target)

Window features are commonly used as inputs to predictive modeling as opposed to the
base feature directly. For example, it is more common practice to use
`did_you_take_drug_x_in_last_30_days` as opposed to `did_you_take_drug_x_that_day`. The
former represents the windowed version, while the latter is the base feature.

Window features tend to work better because:

* Avoid fitting to noise (smoothens a feature)
* Less sparse. Sum of A in the last X days is less sparse than just than just A itself.
  If sum of A is non-zero for 1 row, sum of A in the past 7 days will be non-zero for 7 rows.
* Creates features at the degree of granularity that reflects hypothesis, which are
  usually informed by domain experts.


#### Interacted features
As the name suggests interated features are created when two or more base features are
interacted with each other.

For example:
```
is_x_flag * sum_y
is_x_last_20days * product_is_a
```

Note that `is_x_flag`, `sum_y` and `product_is_a` are considered base features, while
`is_x_last_20_days` is a window feature hence considered a derived feature.

Interacted features may be required because:

* Some models cannot detect interaction. Note that some models can, such as tree-based models.
* Interpretation. If the goal of the model is an interpretable model, then feeding in
  an interacted variable upfront reduces the complexity of applying explainable methods
  on top of models.
