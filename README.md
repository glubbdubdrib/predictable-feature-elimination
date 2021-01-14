# Predictable Feature Elimination

An unsupervised, model-agnostic, wrapper method for performing feature selection. 

We assume that if a feature can be predicted using the others, it adds little information to the problem, and therefore could be removed without impairing the performance of whatever model will be eventually built. The proposed method iteratively identifies and removes such predictable or nearly-predictable redundant feature, allowing to trade-off number of variables and quality. 

The approach do not relies on target labels or values, and whatever model is used to identify predictable features, it is not related to the final use of the feature set. Therefore, it can be used for supervised, unsupervised, or semi-supervised problems, or even as a safe, pre-processing step to improve the quality of the results of other feature selection techniques. Experimental results against state-of-the-art feature-selection algorithms show satisfying performance on several non-trivial benchmarks.

:warning: A paper describing the approach has been submitted to [IJCAI-21](https://ijcai-21.org/).

> PFE is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](http://www.gnu.org/licenses/) as published by the *Free Software Foundation*, either [version 3](https://opensource.org/licenses/GPL-3.0) of the License, or (at your option) any later version.
