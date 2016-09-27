from __future__ import division
from os.path import join

from pyspark import SparkConf, SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkLRLBFGS_mnist8m"
D = 784
K = 10

DATA_DIR = "data/mnist8m"
DATA_FILE_TRAIN = "mnist8m.scale"
DATA_FILE_TEST = "mnist.scale.t"
MODEL_DIR = "stored_model"


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
    conf = SparkConf().setAppName("[TRAIN] {} for MNIST8M".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)

    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN)).map(read_point).cache()
    
    model = LogisticRegressionWithLBFGS.train(data, iterations=50, regType="l2", regParam=0.1, intercept=True, numClasses=K)
    
    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    # Test
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point)
    pred_lab = data.map(lambda x: (int(model.predict(x.features)), int(x.label))).cache()
    acc = pred_lab.filter(lambda (pred, lab): pred == lab).count() / pred_lab.count()
    
    print "\nAccuracy=%0.4f\n" % (acc)
    
    sc.stop()
    