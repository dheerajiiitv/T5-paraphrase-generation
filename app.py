import flask
from flask import Flask, request, render_template
import json
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import subprocess
import time
from translate_models import EnFren
import tensorflow as tf
import nltk.data
flags = tf.flags
FLAGS = flags.FLAGS
import os
# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_string("checkpoint_path", None,
                    "Path to the model checkpoint. Overrides output_dir.")
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                    "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")
flags.FLAGS.problem = "translate_enfr_wmt32k"
flags.FLAGS.model = "transformer"
flags.FLAGS.hparams_set =  "transformer_big"
flags.FLAGS.hparams = "sampling_method=random,sampling_temp=0.7"
flags.FLAGS.decode_hparams  = "beam_size=1,batch_size=16"
flags.FLAGS.checkpoint_path = os.path.join(os.getcwd(),"checkpoints/enfr/model.ckpt-500000")
flags.FLAGS.output_dir =  "/tmp/t2t"
flags.FLAGS.data_dir = os.path.join(os.getcwd(),"checkpoints")


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').eval()
tokenizer = T5Tokenizer.from_pretrained('t5-base')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
app = Flask(__name__)
model_uda = EnFren()
tokenizer_nltk = nltk.data.load('tokenizers/punkt/english.pickle')

def _generate(sentence, num_sentences, max_len, top_p, early_stop):
    text = "paraphrase: " + sentence + " </s>"
    encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
    with torch.no_grad():
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=max_len,
            top_k=100,
            top_p=top_p,
            early_stopping=True if early_stop else False,
            num_return_sequences=num_sentences
        )
    final_outputs = []
    sentence = re.sub(r'[^\w\s]','',sentence)
    for beam_output in beam_outputs:
        sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if sent.lower() != sentence.lower() and sent not in final_outputs:
            final_outputs.append(sent)

    return final_outputs


def _generate_uda(query, samples):
    # Need to define arguments -



    if not model_uda.is_loaded():
        model_uda.load()
    output = []
    for i in range(samples):
        output.append(' '.join(model_uda.predict(query)))


    return output



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_paraphrase', methods=['post'])
def get_paraphrase():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        num_sentences = int(request.json['num_sentences'])
        max_len = int(request.json['max_len'])
        top_p = float(request.json['top_p'])
        is_t5 = bool(request.json['is_t5'])
        is_uda =  bool(request.json['is_uda'])
        early_stop = int(request.json['early_stop'])

        s = time.time()
        if is_t5:
            response = _generate(input_text, num_sentences, max_len, top_p, early_stop)
            str_response = '\n'.join([r for r in response])
        else:
            str_response = ''

        if is_uda:
            uda_response = _generate_uda(tokenizer_nltk.tokenize(input_text), num_sentences)
            uda_response = '\n'.join([r.replace("<EOS>","") for r in uda_response])
        else:
            uda_response = ''

        return app.response_class(response=json.dumps({"t5":str_response, "uda":uda_response}), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001 )

