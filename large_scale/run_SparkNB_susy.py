from __future__ import division
from os.path import join

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkNB_susy"
D = 18

DATA_DIR = "data"
DATA_FILE_TRAIN = "susy_train"
DATA_FILE_TEST = "susy_test"
MODEL_DIR = "stored_model"


def read_point(line):
    items = line.split()
    label = float(items[0]) - 1
        
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
    conf = SparkConf().setAppName("[TRAIN] {} for Susy".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN)).map(read_point).cache()
    
    model = NaiveBayes.train(data, 0.1)
    
    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    # Test
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point)
    
    # Make prediction and test accuracy
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

    # Precision=0.3291, Recall=0.8466, F1=0.4740, Accuracy=0.6657
    print "\nPrecision=%0.4f\tRecall=%0.4f\tF1=%0.4f\tAccuracy=%0.4f\n" % (prec, rec, F1, acc)
    
    sc.stop()
