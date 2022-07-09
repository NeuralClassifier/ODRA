# ODRA: an outlier detection algorithm based on relevant attribute analysis method
Authors: Abdul Wahid and Annavarapu Chandra Sekhara Rao 

Published in Cluster Computing, Springer (2021)

## Abstract
Advances in data acquisition have generated an enormous amount of data that captures business, commercial, technological and scientific information. However, some occurrences are rare or unusual, irrespective of a large amount of data available. These rare occurrences in data mining are usually referred to as outliers or anomalies. All these rare occurrences are infrequent. Sometimes it varies from 0.01% to 10% depending on the type of application. In recent years, outlier detection has become important in many applications and has attracted considerable attention among the increasing number of data mining techniques. Focusing on this has resulted in several outlier detection algorithms, mostly based on distance or density. However, each method has its inherent weaknesses. Methods based on distance have problems with local density, and methods based on density have problems with low-density patterns. In this paper, we present a new outlier detection algorithm based on the relevant attribute analysis (ODRA) for local outlier detection in a high-dimensional dataset. There are two phases of the proposed algorithm. During the preliminary stage, we present a data reduction method that reduces the data set by pruning irrelevant attributes and data points. In the second phase, we propose an outlier detection method based on k-NN kernel density estimation. The experimental results on 15 UCI machine learning repository datasets show the supremacy and effectiveness of our proposed approach over state-of-the-art outlier detection methods.

Paper Link: [Click Here](https://link.springer.com/article/10.1007/s10586-020-03136-9)

## Content

This repository contains the source code.

  * `ODRA.py`: This python file gives only the source code for ODRA.
  
# Instructions
The program is written in Python 3.8:
* Using conda:
```
conda install -c conda-forge jupyterlab
```
* using pip:
```
pip install jupyterlab
```
* Download Python:
```
[Click Here](https://www.python.org/downloads/)
```
## Dependencies
The program requires the following Python libraries:
* scikit-learn v1.0.1
* pandas v1.3.4
* scipy v1.7.3

# Contributors

* Kushankur Ghosh, [kushanku@ualberta.ca](mailto:kushanku@ualberta.ca)

