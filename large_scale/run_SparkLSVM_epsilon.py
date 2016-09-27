from __future__ import division
from os.path import join

import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkLSVM_epsilon"
D = 2000

DATA_DIR = "data"
DATA_FILE_TRAIN = "epsilon_normalized"
DATA_FILE_TEST = "epsilon_normalized.t"
MODEL_DIR = "stored_model"


def read_point(line):
    items = line.split()
    label = int(float(items[0]) == 1)
        
    idx = []
    val = []
    num_items = len(items) - 1
    for j in xrange(num_items):
        u, v = items[j+1].split(":")
        u = int(u)-1
        v = float(v)
        if u < D:
            idx.append(u)
            val.append(v)

    features = SparseVector(D, idx, val)
            
    return LabeledPoint(label, features)


if __name__ == "__main__":
    conf = SparkConf().setAppName("[TRAIN] {} for Epsilon".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    # Load data
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN)).map(read_point).cache()
        
    model = SVMWithSGD.train(data, iterations=100, regType=None, intercept=True)
    
    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    # Test
    model.clearThreshold()
    
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point)
    test_pred_prob = data.map(lambda x: model.predict(x.features)).cache()
    min_prob = test_pred_prob.min()
    max_prob = test_pred_prob.max()
    
            
    for threshold in np.arange(min_prob, max_prob, (max_prob-min_prob)/10):
        # Make prediction and test accuracy.
        model.setThreshold(threshold)
        pred_lab = data.map(lambda x: (int(model.predict(x.features)), int(x.label))).cache()
        TP = pred_lab.filter(lambda (pred, lab): pred & lab).count()
        FN = pred_lab.filter(lambda (pred, lab): pred & (1^lab)).count()
        FP = pred_lab.filter(lambda (pred, lab): (1^pred) & lab).count()
        TN = pred_lab.filter(lambda (pred, lab): (1^pred) & (1^lab)).count()

        eps = 1e-8
        rec = TP / (TP+FN+eps)
        prec = TP / (TP+FP+eps)
        F1 = 2*(rec*prec) / (rec+prec+eps)
        spec = TN / (FP+TN+eps)
        acc = (TP+TN) / (TP+FN+FP+TN)

        # Threshold=0.3299        Precision=0.8079        Recall=0.5397   F1=0.6472       Accuracy=0.5599
        print "\nThreshold=%0.4f\tPrecision=%0.4f\tRecall=%0.4f\tF1=%0.4f\tAccuracy=%0.4f\n" % (threshold, prec, rec, F1, acc)
    
    sc.stop()
