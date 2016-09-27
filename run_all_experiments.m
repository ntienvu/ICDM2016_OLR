%% Standard classification
addpath('standard_classification');
run_Dna_dataset;
run_Covertype_dataset;
run_Mnist_dataset;
run_SenseITVehicle_dataset;
run_KDD_dataset;

%% Label-drift classification
addpath('label_drift');
run_LRSGD_mnist8m_labeldrift;
run_NB_mnist8m_labeldrift;
run_Perceptron_mnist8m_labeldrift;
run_PA_mnist8m_labeldrift;
run_OLR_mnist8m_labeldrift;
