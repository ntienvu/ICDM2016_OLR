One-pass Logistic Regression for Label-drift and Large-scale Classification on Distributed Systems (ICDM 2016 Submission)
===========================

These codes implement One-pass Logistic Regression (OLR) using MATLAB and Apache Spark with Python API, presented in the paper "One-pass Logistic Regression for Label-drift and Large-scale Classification on Distributed Systems" submitted to [ICDM 2016](http://icdm2016.eurecat.org/). The codes are tested on *MATLAB R2014b* and *Apache Spark version 1.4.1, Python 2.7*. Please make sure that you have installed *python-numpy* and you can run Spark on local to run the demo.

1. Run a demo of OLR for standard classification
	
	demo_OLR.m

2. Display a demo of quadrant visualization

	demo_QuadrantVisualization.m

3. Run a demo of OLR on Apache Spark (SparkOLR) for large-scale classification
	
	demo_SparkOLR_large_scale

4. Run a demo of OLR and SparkOLR for label-drift classification
	
	4a. Matlab version of OLR
	addpath('label_drift');
	run_OLR_mnist8m_labeldrift
	
	4b. Spark distributed version of OLR
	demo_SparkOLR_label_drift
	
5. Run all experiments

	run_all_experiments.m
	run_all_spark_experiments

6. To reproduce the results from paper, please download the full datasets (~10G) at the following links, replace the `data` folder:

	http://www.mediafire.com/download/1sdws1lb4h61v2j/icdm16_data.zip.001
	http://www.mediafire.com/download/1fwzunalls4g4tz/icdm16_data.zip.002
	http://www.mediafire.com/download/hya75k9g3kovls5/icdm16_data.zip.003
	http://www.mediafire.com/download/bu4jqn7gejja2t2/icdm16_data.zip.004



### Contact:
Dr Vu Nguyen, vu@ieee.org

### Reference:
Vu Nguyen, Tu Dinh Nguyen, Trung Le, Svetha Venkatesh, and Dinh Phung. "One-Pass Logistic Regression for Label-Drift and Large-Scale Classification on Distributed Systems." In Data Mining (ICDM), 2016 IEEE 16th International Conference on, pp. 1113-1118. IEEE, 2016.
