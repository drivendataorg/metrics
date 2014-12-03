![](https://drivendata.s3.amazonaws.com/images/drivendata.png)

DrivenData Metrics
===============================

Useful implementations of metrics for competitions on [DrivenData](http://www.drivendata.org). 

You can use these as a reference to:
 * Cross-validate your model (highly recommended!) :bowtie:
 * Understand how a metric could be implemented
 * Learn how to use [`numpy`](http://www.numpy.org/)
 * Be annoyed that there are no implementations in R (or Java, Julia, Haskell, Go, Scheme, Fortran, Brainf*ck--we will review pull requests!) :neckbeard: 
 
## Installation
 * Clone the repo or download the raw [`metrics.py`](https://raw.githubusercontent.com/drivendataorg/metrics/master/metrics.py) file.
 * Put the `metrics.py` file in your working directory.

## Usage

Here's a quick example to test predictions in the [Box-plots for Education competition](http://www.drivendata.org/competitions/4/).

```python
import metrics

metrics.multi_multi_log_loss(numpy_array_predictions, 
                             numpy_array_actual_values,
                             metrics.BOX_PLOTS_COLUMN_INDICES)
```

**Disclaimer:** These implementations are not guaranteed to match the implementations in production. There may be numerical differences based on implementation and environment. This code is provided for your convenience only. The only official score is on [DrivenData](http://www.drivendata.org), and the only official description of the metric is on the competition page.
