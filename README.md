
# Gender and Emotion Detection using Audio

## Description

Gender and Emotion recognition by voice is a technique in which you can determine the gender category of a speaker by processing speech signals, in this tutorial, we will be trying to classify gender by voice using the TensorFlow framework in Python.

Gender recognition can be useful in many fields, including automatic speech recognition, in which it can help improve the performance of these systems. It can also be used in categorizing calls by gender, or you can add it as a feature to a virtual assistant that is able to distinguish the talker's gender.

## work flow

Broadly speaking there are two category of features:

### Time domain features
These are simpler to extract and understand, like the energy of signal, zero crossing rate, maximum amplitude, minimum energy, etc.
### Frequency based features
are obtained by converting the time based signal into the frequency domain. Whilst they are harder to comprehend, it provides extra information that can be really handy such as pitch, rhythms, melody etc

MFCC is well known to be a good feature. And there's many ways you can slice and dice this one feature. But what is MFCC? It stands for Mel-frequency cepstral coefficient, and it is a good "representation" of the vocal tract that produces the sound. Think of it like an x-ray of your mouth.

### modeling
The architecture of the model  is based on a few sources that I've seen before such as Kaggle and Stackoverflow. I'm unable to find the source but safe to say this particular format works quite well and is fast, although I've used GPU

## Audio files 

#### Get all Audio files


| Audio files |   Links   |
| :-------- |:-------- |
| Surrey Audio-Visual Expressed Emotion |https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee|
| Ryerson Audio-Visual Database of Emotional Speech and Song |https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio|
| Toronto emotional speech set |https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess|
|Crowd-sourced Emotional Mutimodal Actors Dataset|https://www.kaggle.com/ejlok1/cremad |




## Architecture

```python
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))  # X_train.shape[1] = No. of Columns
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(14)) # Target class number
model.add(Activation('softmax'))
# opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
# opt = keras.optimizers.Adam(lr=0.0001)
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
model.summary()
```


## Accuracy

- Gender at 81% absolute accuracy
- Emotion at 50% absolute accuracy
- Gender and Emotion at 43% absolute accuracy

## Demo



https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenrecording.mkv
## Screenshots

### Image 1:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/7.png?raw=true)
### Image 2:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/6.png?raw=True)
### Image 3:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/5.png?raw=True)
### Image 4:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/4.png?raw=True)
### Image 5:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/3.png?raw=True)
### Image 6:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/2.png?raw=True)
### Image 7:
![App Screenshot](https://github.com/nikhiljanumpally/microsoftinternproject/blob/main/screenshots/1.png?raw=True)

## Azure Services Used

- Azure Machine Learning
- Azure Virtual Machine
- Azure Notebooks
