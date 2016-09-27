from __future__ import division
from os.path import join

from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

MODEL_NAME = "SparkRF_mnist8m"
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
    
    model = RandomForest.trainClassifier(data, numClasses=K, categoricalFeaturesInfo={}, numTrees=30)
    # model = RandomForest.trainRegressor(data, categoricalFeaturesInfo={}, numTrees=30)
    
    # Save learned models
    model.save(sc, join(MODEL_DIR, MODEL_NAME))

    data = sc.textFile(join(DATA_DIR, DATA_FILE_TEST)).map(read_point)
    
    pred = model.predict(data.map(lambda x: x.features))
    pred = pred.map(lambda x: int(x))
    pred_lab = data.map(lambda x: int(x.label)).zip(pred).cache()        
    acc = pred_lab.filter(lambda (lab, pred): pred == lab).count() / pred_lab.count()
        
    # Accuracy=0.8010
    print "\nAccuracy=%0.4f\n" % (acc)
    
    sc.stop()
    