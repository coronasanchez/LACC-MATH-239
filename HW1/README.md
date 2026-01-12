**Before starting, this solution can be run in Colab without changing the ___"Change runtime type"___***
**We can run the solution just in python.**

```
# activate R magic
%load_ext rpy2.ipython
%reload_ext rpy2.ipython
```

Keep **'%%R'** for every code cell ONLY WHEN USING R CODE.
```
%%R
library(readr)
library(ggplot2)
```

Create another code cell and this will allow us to call the libraries in python.
```
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf
import statsmodels.api as sm
import scipy.stats as stats
from IPython.display import display, Math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
