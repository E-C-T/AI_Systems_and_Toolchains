# MQTT IoT: Intrusion Detection

Data Preprocessing and model training with NSL-KDD dataset using 5-fold cross-validation of 2 Spark ML models and 2 TensorFlow models to identify if an attack took place and if so, what attack.

**Constraints and Feature Description**

The following columns were identified to only contain a single unique value, as such, they have been removed from the data frame as not contributing helpful information for the machine learning models to be used after the completion of the MLOps phase.

* "mqtt_conack_flags_reserved", "mqtt_conack_flags_sp", "mqtt_conflag_qos", "mqtt_conflag_reserved", "mqtt_conflag_retain", \
* "mqtt_conflag_willflag", "mqtt_sub_qos", "mqtt_suback_qos", "mqtt_willmsg", "mqtt_willmsg_len", "mqtt_willtopic", \
* "mqtt_willtopic_len"


Of the remaining columns, the TCP columns were identified as follows:
*  tcp_flags: string (nullable = true) (nominal)
*  tcp_time_delta: double (nullable = true) (continuos)
*  tcp_len: integer (nullable = true) (continuous)

Evaluating the Unique values of the MQTT labeled columns Produced the following findings which helped \
to identify likely candidates for binary categorical variables:

mqtt_conack_flags: string (binary) (Need to encode '0x00000000' to 1 and change column to numerical type for binary)

Count of each unique values in train data:
* ('0', 13609885)
* ('0x00000000', 390115)

mqtt_conack_val: double (binary) (conver 5 to 1 for binary)

Count of each unique values in train data:
* (0.0, 13860800)
* (5.0, 139200)

mqtt_conflag_cleansess: double (binary)

Count of each unique values in train data:
* (0.0, 13609360)
* (1.0, 390640)

mqtt_conflag_passwd: double (binary)

Count of each unique values in train data:
* (0.0, 13861600)
* (1.0, 138400)

mqtt_conflag_uname: double (binary)

Count of each unique values in train data:
* (0.0, 13861200)
* (1.0, 138800)

mqtt_conflags: string (nominal)

Count of each unique values in train data:
* ('0', 13609360)
* ('0x00000002', 251840)
* ('0x000000c2', 138400)
* ('0x00000082', 400)

mqtt_dupflag: double (binary)

Count of each unique values in train data:
* (0.0, 13799002)
* (1.0, 200998)

mqtt_hdrflags: string (nominal)

Count of each unique values in train data:
* ('0', 6856327)
* ('0x00000030', 5183544)
* ('0x00000040', 423021)
* ('0x00000010', 390640)
* ('0x00000020', 390115)
* ('0x00000032', 384096)
* ('0x0000003a', 200998)
* ('0x000000d0', 64840)
* ('0x000000c0', 64840)
* ('0x00000031', 15540)
* ('0x00000090', 10367)
* ('0x00000082', 10367)
* ('0x00000050', 5040)
* ('0x000000e0', 265)

mqtt_kalive: double (nominal)

Count of each unique values in train data:
* (0.0, 13609360)
* (60.0, 226595)
* (65535.0, 143325)
* (234.0, 5180)
* (1.0, 5180)
* (3.0, 5180)
* (2.0, 5180)

mqtt_len: double (continuous)

* Contains 92 unique values

mqtt_msg: string (nominal) (Is this useful?, consider droping as it will explode feauture vector)

* Contains 50322 unique values

mqtt_msgid: double (nominal) (Is this useful?, consider droping as it will explode feauture vector)

* Contains 9999 unique values 

mqtt_msgtype: double (already string indexed) ready for OHE -> (nominal)
* 8.0
* 0.0
* 1.0
* 4.0
* 14.0
* 3.0
* 2.0
* 13.0
* 5.0
* 9.0
* 12.0


mqtt_proto_len: double (binary)

Count of each unique values in train data:
* (0.0, 13609360)
* (4.0, 390640)

mqtt_protoname: string (binary) (Need to encode 'MQTT' to 1 and change column to numerical type for binary)

Count of each unique values in train data:
* ('0', 13609360)
* ('MQTT', 390640)

mqtt_qos: double (binary)

Count of each unique values in train data:
* (0.0, 13414906)
* (1.0, 585094)

mqtt_retain: double (binary)

Count of each unique values in train data:
* (0.0, 13984460)
* (1.0, 15540)

mqtt_ver: double (binary)

Count of each unique values in train data:
* (0.0, 13609360)
* (4.0, 390640)



## Machine Learning Modeling 

The task is to predict whether there is an attack and the type of the attack. Since the 'legitimate' label in the target column indicates that there was no attack, I simply evaluated my models based on the accuracy of prediction of the labels in the target column ["legitimate", "slowite", "bruteforce", "flood", "malformed", "dos"].

For SparkML, I used the Random Forest Classifier and Multilayer Perceptron Classifier. 

The Random Forest Classifier had tunable parameters of maximum depth and the number of trees. These directly impact the size and computation time of the model. A model that has many deep trees though is likely to over fit to the train data and perform poorly on the test, preventing it from generalizing well. 

For the Multilayer Perceptron Classifier, the hyper-parameters I chose to tune were the layers and the max iterations. The layer settings specify the depth of each layer, affecting the resolution of the features which can be learned. Large layer settings can capture more variance in the data, but this too can lead to overfitting and longer times to fit the model. Sometimes smaller layer settings are preferred then for better generalization. The max iterations determines how many epochs the model trains for, if the max iterations is too small there is potential that the model has not finished training.

For TensorFlow, I used the 1D-CNN and Multilayer Perceptron

The 1D-CNN had tunable parameters of the number of features for each convolutional layer and the depth of the convolutional block. The number of features for each convolutional layer affects the resolution of the feautres found at a layer as well as the resolution of features found at subsequent layers and directly impacts the size of the model. The depth similarly affects the quality of features learned as the earlier layers learn low-level features and later layers in the convolutional block learn high-level features. More layers though lead to a more complex model with a larger number of parameters.

For the Multilayer Perceptron in Tensorflow my tunable parameters I chose were the depth and width of the neural network. Depth and width both increase the size and complexity of a model by increasing the number of neurons. The relationship between the two is not very clear. Small depth and small width models tend to perform poorly and large depth and large width models tend to overfit, so it is often that a hyperparameter search is performed with these parameters to find the best combination.

I chose the models that I did since I know they perform well with high dimenionality data by finding non-linear mappings between input and output and can be used for multiclass classification.

Comparing the models, Random Forest is an ensemble machine learning algorithm that works on structured data, while both Multilayer Perceptron and CNN are deep learning methods which do not require feature engineering before hand, as the "features" are learned by the networks. However, the quality and cleanliness of the data can impact the quality of the "features" learned by deep learning models. Deep learning models typically require more data, especially in the case of binary classification when compared to Random Forest presented with data that has undergone good feature engineering. The primary difference between convolutional layers in a CNN and dense layers in an MLP is that a convolutional layer perform convolutions to find features based on a sliding window and the layers are not fully connected, while Multilayer Perceptrons are densly connected and features are more ephemeral encodings of the relationship between input and output based the weights and biases between neurons.

In general, I found that the models performed in the following order from best to worst: SparkML Random Forest Classifier, TensorFlow 1D-CNN, TensorFlow Multilayer Perceptron, SparkML Multilayer Perceptron. It is important to note though, since the Multilayer Perceptrons of the Spark ML and TensorFlow implementations used different hyperparameter settings, it is not a fair comparison between the two.


### Tensorboard Parallel Coordinates View for Tensorflow 1D-CNN (Local)
![cnn_tb](./cnn_tb.png?raw=true)

### Tensorboard Parallel Coordinates View for Tensorflow MLP (Local)
![mlp_tb](./mlp_tb.png?raw=true)

The log files associated with the hyperparameter search for the TensorFlow models on the local machine can be found in the *logs/* folder.

## On the Cloud:

The Google Cloud implementation can be found in *MQTT_IoT_Intrusion_Detection_Cloud_Run.ipynb*



