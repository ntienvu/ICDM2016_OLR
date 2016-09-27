from __future__ import division
from os.path import join

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkNB_airlines"

DATA_DIR = "data"
DATA_FILE_TRAIN = "airlines_1987_to_2008_train90"
DATA_FILE_TEST = "airlines_1987_to_2008_test90"
MODEL_DIR = "stored_model"

D = 857
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
    conf = SparkConf().setAppName("[TRAIN] NaiveBayes for Airline Delay Prediction")
    sc = SparkContext(conf = conf)
    
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN)).map(read_point).cache()
    model = NaiveBayes.train(data, 0.1)

    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point)
    
    # Make prediction and test accuracy
    pred_lab = data.map(lambda x: (int(model.predict(x.features)), int(x.label))).cache()
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

    # Precision=0.09, Recall=0.34, F1=0.14, Accuracy=0.82
    print "\nPrecision=%0.2f, Recall=%0.2f, F1=%0.2f, Accuracy=%0.2f\n" % (prec, rec, F1, acc)
    
    sc.stop()
