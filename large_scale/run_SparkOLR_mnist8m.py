from __future__ import division
import time
from os.path import join
import numpy as np
from operator import add
from pyspark import SparkConf, SparkContext
import cPickle as pkl

MODEL_NAME = "SparkOLR_mnist8m"
D = 784
K = 10
NUM_TRAIN_PARTITIONS = 800
NUM_TEST_PARTITIONS = 1

DATA_DIR = "data/mnist8m"
DATA_FILE_TRAIN = "mnist8m.scale"
DATA_FILE_TEST = "mnist.scale.t"
MODEL_DIR = "model"


def read_point_batch(iterator):
    lines = list(iterator)
    y = np.zeros(len(lines), dtype=np.int)
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        items = lines[i].split()
        y[i] = np.int(np.float(items[0]))
        num_items = len(items) - 1
        for j in xrange(num_items):
            u, v = items[j+1].split(":")
            X[i, np.int(u)-1] = np.float(v)
            
    return [(X, y)]


def scale_data(X, y):
    y_label = sorted(np.unique(y))
    y_count = np.array([len(np.where(y==i)[0]) for i in y_label])
    y_count = (y_count.min()+1e-8) / (y_count+1e-8)
    X *= y_count[y].reshape(len(y), 1)
    return (X, y)

    
def sample_polygam(X, y, w):
    wx = (w[y, :]*X).sum(axis=1, keepdims=True)
    return (X, y, np.tanh(wx / 4.0) / wx)


def calc_q(X, y, invlb):
    q = np.zeros((K, D))
    for i in xrange(K):
        q[i, :] = X[y==i, :].sum(axis=0, keepdims=True) - X[y!=i, :].sum(axis=0, keepdims=True)
    return q


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
    conf = SparkConf().setAppName("[TRAIN] {} for MNIST8M".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    
    start_time = time.time()
    #data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch).map(lambda batch: scale_data(batch[0], batch[1]))
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch)
    
    print data.map(lambda batch: np.sum(batch[0], axis=0)).reduce(add)
    
    print data.map(lambda batch: np.sum(batch[0], axis=1).max()).max()
    print data.map(lambda batch: np.sum(batch[0], axis=1).sum()).reduce(add) / 8100000.0
    print data.map(lambda batch: np.tanh(np.sum(batch[0], axis=1) / 4).min()).min()
    print data.map(lambda batch: np.tanh(np.sum(batch[0], axis=1) / 4).max()).max()
    print data.map(lambda batch: np.tanh(np.sum(batch[0], axis=1) / 4).sum()).sum() / 8100000.0
    
#     sum_y = data.map(lambda batch: np.sum(batch[1])).reduce(add)
#     print sum_x, sum_y

    w = 1.0*np.ones((K, D))
    data = data.map(lambda batch: sample_polygam(batch[0], batch[1], w))
    
    sum_lb = data.map(lambda batch: np.sum(batch[2])).reduce(add)
    print sum_lb

    p = np.eye(D) + data.map(lambda batch: (batch[0]*batch[2]).T.dot(batch[0])).reduce(add)

    q = data.map(lambda batch: calc_q(batch[0], batch[1], batch[2])).reduce(add)

    for i in xrange(K):
        w[i, :] = np.linalg.solve(p, q[i, :])
        
    print "Time %.2f second(s)" % (time.time() - start_time)
        
    # Save learned models
    pkl.dump({"w": w, "p": p, "q": q}, open(join(MODEL_DIR, MODEL_NAME+".pkl"), "wb"))
    
    # Load model
    w = pkl.load(open(join(MODEL_DIR, MODEL_NAME+".pkl"), "rb"))['w']
    
    data_test = sc.textFile(join(DATA_DIR, DATA_FILE_TEST), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch)
    
    # Make prediction and test accuracy.
    pred_lab = data_test.map(lambda batch: (batch[0].dot(w.T).argmax(axis=1), batch[1])).cache()
    acc = pred_lab.map(lambda (pred, lab): (pred == lab).sum()).sum() / pred_lab.map(lambda (pred, lab): len(pred)).sum()
    
    # Accuracy=0.
    print "\nAccuracy=%0.4f\n" % (acc)
    
    sc.stop()
