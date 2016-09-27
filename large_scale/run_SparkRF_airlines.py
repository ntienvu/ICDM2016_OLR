from __future__ import division
from os.path import join

import numpy as np

from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkRF_airlines"
D = 857

DATA_DIR = "data"
DATA_FILE_TRAIN = "airlines_1987_to_2008_train90"
DATA_FILE_TEST = "airlines_1987_to_2008_test90"
MODEL_DIR = "stored_model"

MAX_DISTANCE = 4983.0


def read_point(line):
    items = line.split()
    label = float(items[0])
        
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
    conf = SparkConf().setAppName("[TRAIN] {} for Airline Delay Prediction".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN)).map(read_point).cache()
    
    # model = RandomForest.trainClassifier(data, numClasses=2, categoricalFeaturesInfo={}, numTrees=30)
    model = RandomForest.trainRegressor(data, categoricalFeaturesInfo={}, numTrees=30)
    
    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    # Load data
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point).cache()
    
    pred_prob = model.predict(data.map(lambda x: x.features))
    min_prob = pred_prob.min()
    max_prob = pred_prob.max()
                
    for threshold in np.arange(min_prob, max_prob, (max_prob-min_prob)/10):
        # Make prediction and test accuracy.
        pred = pred_prob.map(lambda x: int(x >= threshold))
        pred_lab = data.map(lambda x: int(x.label)).zip(pred).cache()
        
        TP = pred_lab.filter(lambda (pred, lab): pred & lab).count()
        FN = pred_lab.filter(lambda (pred, lab): pred & (1^lab)).count()
        FP = pred_lab.filter(lambda (pred, lab): (1^pred) & lab).count()
        TN = pred_lab.filter(lambda (pred, lab): (1^pred) & (1^lab)).count()

        eps = 1e-8
        rec = TP / (TP+FN+eps);
        prec = TP / (TP+FP+eps);
        F1 = 2*(rec*prec) / (rec+prec+eps);
        spec = TN / (FP+TN+eps);
        acc = (TP+TN) / (TP+FN+FP+TN)

        # Threshold=0.17, Precision=0.19, Recall=0.89, F1=0.31, Accuracy=0.35
        print "\nThreshold=%0.2f, Precision=%0.2f, Recall=%0.2f, F1=%0.2f, Accuracy=%0.2f\n" % (threshold, prec, rec, F1, acc)
    
    sc.stop()
    