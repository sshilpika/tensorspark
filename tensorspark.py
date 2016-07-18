import parameterwebsocketclient
import pyspark
from operator import add
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import os
import mnistdnn
import higgsdnn
import moleculardnn
import tensorflow as tf
import time
import random
import cStringIO
import numpy as np
#from memory_profiler import profile
#import sys

directory = "/projects/ExaHDF5/sshilpika/tensorspark/"

model_keyword = 'mnist'
if model_keyword == 'mnist':
    training_rdd_filename = '%stiny_mnist_train.csv' % directory
    test_filename = '%stiny_mnist_test.csv' % directory
    local_test_path = '/scratch/tiny_mnist_test.csv'
    partitions = 48
    warmup = 2000
    batch_sz = 50
    epochs = 5
    repartition = True
    time_lag = 100
    model = mnistdnn.MnistDNN(batch_sz)
elif model_keyword == 'higgs':
    training_rdd_filename = '%shiggs_train_all.csv' % directory
    test_filename = '%shiggs_test_all.csv' % directory
    local_test_path = '/scratch/higgs_test_all.csv'
    warmup = 20000
    epochs = 1
    partitions = 64
    batch_sz = 128
    time_lag = 20
    repartition = True
    model = higgsdnn.HiggsDNN(batch_sz)
elif model_keyword == 'molecular':
    training_rdd_filename = '%smolecular_train_all.csv' % directory
    test_filename = '%smolecular_test_all.csv' % directory
    local_test_path = '/scratch/molecular_test_all.csv'
    warmup = 10000
    repartition = True
    epochs = 3
    partitions = 128
    batch_sz = 64
    time_lag = 130
    model = moleculardnn.MolecularDNN(batch_sz)
else:
    print("KEYWORD HAS TO BE 'mnist', 'higgs' or 'molecular'")
    sys.exit(1)

t = int(time.time())
error_rates_path = '/home/ubuntu/error_rates_%s_%d.txt' % (model_keyword, t)
conf = pyspark.SparkConf()
#conf.setMaster('yarn')
#conf.set('spark.driver.memory', '14g')
#conf.set('spark.executor.memory', '8g')
#conf.set('spark.driver.maxResultSize', '14g')
#conf.set('spark.yarn.am.memory', '10g')
#conf.set('yarn.nodemanager.resource.memory-mb', '2000')
conf.setExecutorEnv('LD_LIBRARY_PATH', ':/soft/visualization/cuda-7.5.18/lib64')
conf.setExecutorEnv('PATH', '/soft/libraries/mpi/mvapich2-2.1/intel/bin:/soft/compilers/gcc/4.9.3/bin:/soft/libraries/anaconda/bin:/soft/visualization/cuda-7.5.18/bin:/soft/compilers/java/jdk1.8.0_60/bin:/soft/environment/softenv-1.6.2/bin:/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/X11R6/bin:/soft/buildtools/trackdeps/bin:/dbhome/db2cat/sqllib/bin:/dbhome/db2cat/sqllib/adm:/dbhome/db2cat/sqllib/misc:/usr/lpp/mmfs/bin:/opt/ibutils/bin')
#conf.setExecutorEnv('HADOOP_CONF_DIR', '/usr/local/hadoop/etc/hadoop')
conf.setExecutorEnv('JAVA_HOME','/soft/compilers/java/jdk1.8.0_60')
sc = pyspark.SparkContext(conf=conf)

websocket_port = random.randint(30000, 60000)
print 'websocket_port %d' % websocket_port
class ParameterServerWebsocketHandler(tornado.websocket.WebSocketHandler):

        def __init__(self, *args, **kwargs):
                self.server = kwargs.pop('server')
                self.model = self.server.model
                with self.model.session.graph.as_default():
                        self.saver = tf.train.Saver()
                self.lock = threading.Lock()
                super(ParameterServerWebsocketHandler,self).__init__(*args, **kwargs)

        def open(self):
                self.send_parameters()

        def send_parameters(self):
                self.lock.acquire()
                parameters = self.model.get_parameters()
                self.lock.release()
                serialized = self.model.serialize(parameters)
                self.write_message(serialized, binary=True)

        def on_close(self):
                pass


        def on_message(self, message):
                # now assuming every message is a gradient
                time_gradient = self.model.deserialize(message)
                self.server.gradient_count += 1
                print 'gradient_count %d' % self.server.gradient_count
                time_sent = time_gradient[0][0]
                #print(time.time() - time_sent)
                if time.time() - time_sent < time_lag:
                        self.lock.acquire()
                        gradient = time_gradient[1:]
                        self.model.apply(gradient)
                        if self.server.gradient_count % 10 == 0:
                               error_rate = self.model.test(self.server.test_labels, self.server.test_features)
                               print 'gradients received: %d    error_rate: %f' % (self.server.gradient_count, error_rate)
                               t = time.time()
                               with open(error_rates_path, 'a') as f:
                                       f.write('%f, %d, %f\n' % (t, self.server.gradient_count, error_rate))

                        self.lock.release()
                else:
                        print "Rejected"
                del time_gradient
                self.send_parameters()

class ParameterServer(threading.Thread):

        def __init__(self, model, warmup_data=None, test_data=None):
                threading.Thread.__init__(self)
                self.model = model
                test_labels, test_features = model.process_data(test_data)
                self.test_features = test_features
                self.test_labels = test_labels
                self.warmup(warmup_data)
                self.gradient_count = 0
                self.application = tornado.web.Application([(r"/", ParameterServerWebsocketHandler, {'server':self})])

        def warmup(self, data=None):
                if data is not None:
                        self.model.train_warmup(partition=data, error_rates_filename=error_rates_path)

        def run(self):
                self.application.listen(websocket_port)
                tornado.ioloop.IOLoop.current().start()


def train_partition(partition):
        return parameterwebsocketclient.TensorSparkWorker(model_keyword, batch_sz, websocket_port).train_partition(partition)

def test_partition(partition):
        return parameterwebsocketclient.TensorSparkWorker(model_keyword, batch_sz, websocket_port).test_partition(partition)

# you can find the mnist csv files here http://pjreddie.com/projects/mnist-in-csv/
def train_epochs(num_epochs, training_rdd, num_partitions):
        for i in range(num_epochs):
                print 'training epoch %d' % i
                if repartition:
                        training_rdd = training_rdd.repartition(num_partitions)
                mapped_training = training_rdd.mapPartitions(train_partition)
                mapped_training.collect()
                #training_rdd.repartition(training_rdd.getNumPartitions())


def test_all():
        testing_rdd = sc.textFile(test_filename).cache()
        #testing_rdd = sc.textFile('%shiggs_test_all.csv' % directory)
        #testing_rdd = sc.textFile('%smolecular_test_all.csv' % directory)
        mapped_testing = testing_rdd.mapPartitions(test_partition)
        return mapped_testing.reduce(add)/mapped_testing.getNumPartitions()




def start_parameter_server(model, warmup_data,test_data):
        parameter_server = ParameterServer(model=model, warmup_data=warmup_data, test_data=test_data)
        parameter_server.start()
        return parameter_server


def main(warmup_iterations, num_epochs, num_partitions):
        try:
                training_rdd = sc.textFile(training_rdd_filename, minPartitions=num_partitions).cache()
                print 'num_partitions = %s' % training_rdd.getNumPartitions()
                time.sleep(5)

                warmup_data = training_rdd.take(warmup_iterations)

                with open(local_test_path) as test_file:
                        test_data_lines = test_file.readlines()

                with open(error_rates_path, 'w') as f:
                        f.write('')
                test_data = test_data_lines[0:100]

                parameter_server = start_parameter_server(model=model, warmup_data=warmup_data, test_data=test_data)
                #raw_input('Press enter to continue\n')

                #training_rdd = training_rdd.subtract(sc.parallelize(warmup_data))
                train_epochs(num_epochs, training_rdd, num_partitions)
#               save_model()
#                test_results = test_all()
#               sc.show_profiles()
#                t = time.time()
#                with open(error_rates_path, 'a') as f:
#                        f.write('%f , %f\ndone' % (t, test_results))
#                print test_results
                print 'done'
#                return test_results
        finally:
                tornado.ioloop.IOLoop.current().stop()

main(warmup_iterations=warmup, num_epochs=epochs, num_partitions=partitions)
