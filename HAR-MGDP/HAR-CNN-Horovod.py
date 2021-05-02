import matplotlib

matplotlib.use('Agg')
import pandas as pd
from IPython.display import display
import math
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.compat.v1.keras import backend as K
import horovod.tensorflow.keras as hvd
from tensorflow.keras.models import Sequential
# Horovod: initialize Horovod.
hvd.init()
# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.compat.v1.Session(config=config))

from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
import sys 
import time

# In[2]:


def load_activity_map():
    map = {}
    map[0] = 'transient'
    map[1] = 'lying'
    map[2] = 'sitting'
    map[3] = 'standing'
    map[4] = 'walking'
    map[5] = 'running'
    map[6] = 'cycling'
    map[7] = 'Nordic_walking'
    map[9] = 'watching_TV'
    map[10] = 'computer_work'
    map[11] = 'car driving'
    map[12] = 'ascending_stairs'
    map[13] = 'descending_stairs'
    map[16] = 'vacuum_cleaning'
    map[17] = 'ironing'
    map[18] = 'folding_laundry'
    map[19] = 'house_cleaning'
    map[20] = 'playing_soccer'
    map[24] = 'rope_jumping'
    return map


# In[3]:


def generate_three_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    return [x, y, z]


def generate_four_IMU(name):
    x = name + '_x'
    y = name + '_y'
    z = name + '_z'
    w = name + '_w'
    return [x, y, z, w]


def generate_cols_IMU(name):
    # temp
    temp = name + '_temperature'
    output = [temp]
    # acceleration 16
    acceleration16 = name + '_3D_acceleration_16'
    acceleration16 = generate_three_IMU(acceleration16)
    output.extend(acceleration16)
    # acceleration 6
    acceleration6 = name + '_3D_acceleration_6'
    acceleration6 = generate_three_IMU(acceleration6)
    output.extend(acceleration6)
    # gyroscope
    gyroscope = name + '_3D_gyroscope'
    gyroscope = generate_three_IMU(gyroscope)
    output.extend(gyroscope)
    # magnometer
    magnometer = name + '_3D_magnetometer'
    magnometer = generate_three_IMU(magnometer)
    output.extend(magnometer)
    # oreintation
    oreintation = name + '_4D_orientation'
    oreintation = generate_four_IMU(oreintation)
    output.extend(oreintation)
    return output


def load_IMU():
    output = ['time_stamp', 'activity_id', 'heart_rate']
    hand = 'hand'
    hand = generate_cols_IMU(hand)
    output.extend(hand)
    chest = 'chest'
    chest = generate_cols_IMU(chest)
    output.extend(chest)
    ankle = 'ankle'
    ankle = generate_cols_IMU(ankle)
    output.extend(ankle)
    return output


def load_subjects(root='input/ass2-time-series/PAMAP2_Dataset/Protocol/subject'):
    output = pd.DataFrame()
    cols = load_IMU()
    for i in range(101, 110):
    #for i in range(101, 103):
        path = root + str(i) + '.dat'
        subject = pd.read_table(path, header=None, sep='\s+')
        subject.columns = cols
        subject['id'] = i
        output = output.append(subject, ignore_index=True)
    output.reset_index(drop=True, inplace=True)
    return output


data = load_subjects()

def fix_data(data):
    data = data.drop(data[data['activity_id'] == 0].index)
    data = data.interpolate()
    # fill all the NaN values in a coulmn with the mean values of the column
    for colName in data.columns:
        data[colName] = data[colName].fillna(data[colName].mean())
    activity_mean = data.groupby(['activity_id']).mean().reset_index()
    return data


data = fix_data(data)


# In[6]:
print('Size of the data: ', data.size)
print('Shape of the data: ', data.shape)
print('Number of columns in the data: ', len(data.columns))
result_id = data.groupby(['id']).mean().reset_index()
print('Number of uniqe ids in the data: ', len(result_id))
result_act = data.groupby(['activity_id']).mean().reset_index()
print('Numbe of uniqe activitys in the data: ', len(result_act))


# In[7]:
def pd_fast_plot(pd, column_a, column_b, title, figsize=(10, 6)):
    plt.rcParams.update({'font.size': 16})
    size = range(len(pd))
    f, ax = plt.subplots(figsize=figsize)
    plt.bar(size, pd[column_a], color=plt.cm.Paired(size))
    a = ax.set_xticklabels(pd[column_b])
    b = ax.legend(fontsize=20)
    c = ax.set_xticks(np.arange(len(pd)))
    d = ax.set_title(title)
    plt.show()


# In[8]:
sampels = data.groupby(['id']).count().reset_index()
sampels_to_subject = pd.DataFrame()
sampels_to_subject['id'] = sampels['id']
sampels_to_subject['sampels'] = sampels['time_stamp']
sampels_to_subject = sampels_to_subject.sort_values(by=['sampels'])
pd_fast_plot(sampels_to_subject, 'sampels', 'id', 'Number Of Samepls By Users')
# In[9]:


map_ac = load_activity_map()
sampels = data.groupby(['activity_id']).count().reset_index()
sampels_to_subject = pd.DataFrame()
sampels_to_subject['activity'] = [map_ac[x] for x in sampels['activity_id']]
sampels_to_subject['sampels'] = sampels['time_stamp']
sampels_to_subject = sampels_to_subject.sort_values(by=['sampels'])
pd_fast_plot(sampels_to_subject, 'sampels', 'activity', 'Number Of Samepls By Activity', figsize=(40, 7))

# In[10]:


sampels_heart_rate = pd.DataFrame()
sampels_heart_rate['id'] = result_id['id']
sampels_heart_rate['heart_rate'] = result_id['heart_rate']
sampels_heart_rate = sampels_heart_rate.sort_values(by=['heart_rate'])
pd_fast_plot(sampels_heart_rate, 'heart_rate', 'id', 'Avg heart Rate by Subject')

# In[11]:


map_ac = load_activity_map()
sampels_heart_rate = pd.DataFrame()
sampels_heart_rate['activity'] = [map_ac[x] for x in result_act['activity_id']]
sampels_heart_rate['heart_rate'] = result_act['heart_rate']
sampels_heart_rate = sampels_heart_rate.sort_values(by=['heart_rate'])
pd_fast_plot(sampels_heart_rate, 'heart_rate', 'activity', 'Avg heart Rate by Activity', figsize=(40, 10))

# In[12]:

# Data Spliting and Test
def split_train_test(data):
    # create the test data
    subject107 = data[data['id'] == 107]
    subject108 = data[data['id'] == 108]
    test = subject107.append(subject108)

    # create the train data
    train = data[data['id'] != 107]
    train = data[data['id'] != 108]

    # drop the columns id and time
    test = test.drop(["id"], axis=1)
    train = train.drop(["id"], axis=1)

    # split train and test to X and y
    X_train = train.drop(['activity_id', 'time_stamp'], axis=1).values
    X_test = test.drop(['activity_id', 'time_stamp'], axis=1).values

    # make data scale to min max beetwin 0 to 1
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(X_train)
    min_max_scaler.fit(X_test)
    X_train = min_max_scaler.transform(X_train)
    X_test = min_max_scaler.transform(X_test)

    y_train = train['activity_id'].values
    y_test = test['activity_id'].values
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = split_train_test(data)
print('Train shape X :', X_train.shape, ' y ', y_train.shape)
print('Test shape X :', X_test.shape, ' y ', y_test.shape)


# Assignment of batch size
if len(sys.argv) != 2:
    batch_size = 500
    print(f'By Default Batch Size is {batch_size}')
       #By default predefined batch size in case if user doesn't send one
else:
    batch_size = sys.argv[1]
    print(f'User Defined Batch Size is {batch_size}')
#Reading from passed command line arguments
#batch_size = 100
batch_size = int(batch_size)

# In[13]:


def create_cnn_data(X, y, step_back=50, step_forword=1):
    out_X = []
    out_y = []
    size = len(X)
    for i, features in enumerate(X):
        if i >= step_back and i < size - step_forword:
            tmp_X = []
            tmp_y = []
            for j in range(i - step_back, i):
                tmp_X.extend([X[j]])
            out_X.append(tmp_X)
            for j in range(i, i + step_forword):
                tmp_y.extend([y[j]])
            out_y.append(tmp_y)
    #print('Out side loop iteration: %d ' % (i))
    return np.array(out_X), np.array(out_y)


X_cnn_train, y_cnn_train = create_cnn_data(X_train, y_train)
X_cnn_test, y_cnn_test = create_cnn_data(X_test, y_test)

# In[ ]:
# hot encoded
hot = OneHotEncoder(handle_unknown='ignore', sparse=False)
hot.fit(y_cnn_train)
hot.fit(y_cnn_test)
y_h_train = hot.transform(y_cnn_train)
y_h_test = hot.transform(y_cnn_test)

# In[25]:

###################################################3

 
   
               
if hvd.rank() == 0:
    print('Number of GPU: %d' % (hvd.size()))

# Horovod: pin GPU to be used to process local rank (one GPU per process)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    
    
num_classes = 12

# Horovod: adjust number of epochs based on number of GPUs.
epochs = int(math.ceil(12.0 / hvd.size()))

# Input dimensions
rows, cols = 50, 52

x_train = X_cnn_train.reshape(X_cnn_train.shape[0], rows, cols, 1)
x_test = X_cnn_test.reshape(X_cnn_test.shape[0], rows, cols, 1)
input_shape = (rows, cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


opt = tf.keras.optimizers.Adam(0.00001 * hvd.size())

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1)

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))
    
start_time = time.time()
model.fit(
    x_train,
    y_h_train,
    batch_size=batch_size,
    callbacks=callbacks,
    epochs=epochs,
    verbose=1 if hvd.rank() == 0 else 0,
)

print("Trainning Time: %s seconds " % (time.time() - start_time))
score = model.evaluate(x_test, y_h_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])