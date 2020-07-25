# Paraphrase generation using T5 model
Simple application using T5 base model fine tuned in Quora Question Pairs to generate paraphased questions.

This repository is based on the work from [@ramsrigouthamg](https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer-) which explain very well how to fine tune the model.

# Paraphrase generation using Google UDA.
You will able to create paraphrase using a technique call ** Back Translate **. 


### Application

![Paraphrase](paraphrase.gif)


### Install
1. 
```
pip install -r requirements.txt
```
2. replace ``` core_estimator_predictor.py``` in ``` your_env/lib/python3.6/site-packages/tensorflow/contrib/predictor ``` with the file in given folder
3. Download models
```wget https://storage.googleapis.com/uda_model/text/back_trans_checkpoints.zip```
```unzip back_trans_checkpoints.zip && rm back_trans_checkpoints.zip```



### Running 

```
cd web-app
python app.py
```

Open your browser http://localhost:8001


