from __future__ import division
import time
from os.path import join
import numpy as np
from operator import add
from pyspark import SparkConf, SparkContext
import cPickle as pkl

MODEL_NAME = "SparkOLR_epsilon"
D = 2000
K = 2
NUM_TRAIN_PARTITIONS = 100
NUM_TEST_PARTITIONS = 25

DATA_DIR = "data"
DATA_FILE_TRAIN = "epsilon_normalized"
DATA_FILE_TEST = "epsilon_normalized.t"
MODEL_DIR = "model"


def read_point_batch(iterator):
    lines = list(iterator)
    y = np.zeros(len(lines), dtype=np.int)
    X = np.zeros((len(lines), D))
    for i, line in enumerate(lines):
        items = lines[i].split()
        y[i] = int(np.int(np.float(items[0])) == 1)
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
        q[i, :] = 0.5*X[y==i, :].sum(axis=0, keepdims=True) - 0.5*X[y!=i, :].sum(axis=0, keepdims=True)
    return q


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
    conf = SparkConf().setAppName("[TRAIN] {} for Epsilon".format(MODEL_NAME))
    sc = SparkContext(conf = conf)

    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org").setLevel(logger.Level.ERROR)
    logger.LogManager.getLogger("akka").setLevel(logger.Level.ERROR)
    
    start_time = time.time()
    data = sc.textFile(join(DATA_DIR, DATA_FILE_TRAIN), NUM_TRAIN_PARTITIONS).mapPartitions(read_point_batch)

    w = 1.0*np.ones((K, D))
    data = data.map(lambda batch: sample_polygam(batch[0], batch[1], w))

    p = np.eye(D) + data.map(lambda batch: (batch[0]*batch[2]).T.dot(batch[0])).reduce(add)

    q = data.map(lambda batch: calc_q(batch[0], batch[1], batch[2])).reduce(add)

#     mm = max(p.max(), q.max())
#     p /= mm
#     q /= mm    
    
    for i in xrange(K):
        w[i, :] = np.linalg.solve(p, q[i, :])
        
    print "Time %.2f second(s)" % (time.time() - start_time)
        
    # Save learned models
    pkl.dump({"w": w, "p": p, "q": q}, open(join(MODEL_DIR, MODEL_NAME+".pkl"), "wb"))
    
    # Load model
    w = pkl.load(open(join(MODEL_DIR, MODEL_NAME+".pkl"), "rb"))['w']
    
    data_test = sc.textFile(join(DATA_DIR, DATA_FILE_TEST), NUM_TEST_PARTITIONS).mapPartitions(read_point_batch)
    
    w_bc = sc.broadcast(w)

    # Make prediction and test accuracy.
    pred_lab = data_test.map(lambda batch: (batch[0].dot(w_bc.value.T).argmax(axis=1), batch[1])).cache()
    TP = pred_lab.map(lambda (pred, lab): (pred & lab).sum()).sum()
    FN = pred_lab.map(lambda (pred, lab): (pred & (1^lab)).sum()).sum()
    FP = pred_lab.map(lambda (pred, lab): ((1^pred) & lab).sum()).sum()
    TN = pred_lab.map(lambda (pred, lab): ((1^pred) & (1^lab)).sum()).sum()    
    
    # Make prediction and test accuracy.
    eps = 1e-8
    rec = TP / (TP+FN+eps)
    prec = TP / (TP+FP+eps)
    F1 = 2*(rec*prec) / (rec+prec+eps)
    spec = TN / (FP+TN+eps)
    acc = (TP+TN) / (TP+FN+FP+TN)
    
    print "\nprecision = %0.4f, recall = %0.4f, F1 = %0.4f, accuracy = %0.4f\n" % (prec, rec, F1, acc)
    
    sc.stop()
