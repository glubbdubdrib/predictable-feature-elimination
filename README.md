# Predictable Feature Elimination

An unsupervised, model-agnostic, wrapper method for performing feature selection. 

We propose an unsupervised, model-agnostic, wrapper method for feature selection. We assume that if a feature can be predicted using the others, it adds little information to the problem, and therefore could be removed without impairing the performance of whatever model will be eventually built. The proposed method iteratively identifies and removes predictable, or nearly-predictable, redundant features, allowing to trade-off complexity with expected quality. 

The approach do not rely on target labels nor values, and the model used to identify predictable features is not related to the final use of the feature set. Therefore, it can be exploited for supervised, unsupervised, or semi-supervised problems, or even as a safe, pre-processing step to improve the quality of the results of other feature selection techniques.

> **Copyright (c) 2020 [Pietro Barbiero](https://github.com/pietrobarbiero), [Giovanni Squillero](https://github.com/squillero), and [Alberto Tonda](https://github.com/albertotonda)**  
> This program is free software: you can redistribute it and/or modify it under the terms of the [GNU General Public License](http://www.gnu.org/licenses/) as published by the *Free Software Foundation*, either [version 3](https://opensource.org/licenses/GPL-3.0) of the License, or (at your option) any later version.
