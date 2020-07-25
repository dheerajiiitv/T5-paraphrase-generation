import  tensorflow as tf
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import text_encoder
import os
import numpy as np
FLAGS = tf.flags.FLAGS

class EnFren():
    def __init__(self):
        self.hparams_set = FLAGS.hparams_set
        self.hparams = FLAGS.hparams
        self.decode_hparams = FLAGS.decode_hparams
        self.data_dir = FLAGS.data_dir
        self.problem = FLAGS.problem
        self.decode_shards = 1
        self.worker_id = 0
        self.decode_in_memory = False
        self.model = FLAGS.model
        self.use_tpu = False
        self.checkpoint_dir = FLAGS.checkpoint_path
        self.estimator = None
        self.estimator_predictor = None
        self.hp = self.create_hparams()
        self.decode_hp = self.create_decode_hparams()
        self.p_hp = self.hp.problem_hparams
        self.has_input = "inputs" in self.p_hp.vocabulary
        self.inputs_vocab_key = "inputs" if self.has_input else "targets"
        self.inputs_vocab = self.p_hp.vocabulary[self.inputs_vocab_key]
        self.targets_vocab = self.p_hp.vocabulary["targets"]
        self.problem_name = self.problem
        self.estimator_decoder_predictor = None

    def load(self):

        run_config = t2t_trainer.create_run_config(self.hp)
        self.hp.add_hparam("model_dir", run_config.model_dir)
        self.estimator = trainer_lib.create_estimator(
            self.model,
            self.hp,
            run_config,
            decode_hparams=self.decode_hp,
            use_tpu=self.use_tpu)

        self.estimator_predictor = tf.contrib.predictor.from_estimator(self.estimator, self.input_fn,
                                                                  config=tf.ConfigProto(log_device_placement=True,
                                                                                  allow_soft_placement=True))
        FLAGS.problem = "translate_enfr_wmt32k_rev"
        self.problem = "translate_enfr_wmt32k_rev"
        self.problem_name = self.problem
        FLAGS.checkpoint_path = os.path.join(os.getcwd(),"checkpoints/fren/model.ckpt-500000")
        run_config = t2t_trainer.create_run_config(self.hp)
        self.hp.model_dir = run_config.model_dir
        self.estimator = trainer_lib.create_estimator(
            self.model,
            self.hp,
            run_config,
            decode_hparams=self.decode_hp,
            use_tpu=self.use_tpu)

        self.estimator_decoder_predictor = tf.contrib.predictor.from_estimator(self.estimator, self.input_fn,
                                                                       config=tf.ConfigProto(log_device_placement=True,
                                                                                             allow_soft_placement=True))




    def is_loaded(self):
        if self.estimator_predictor and self.estimator:
            return True
        else:
            return False

    def predict(self, query):
        preprocessed_input = self.preprocess_input(query)
        preprocessed_input = self.estimator_predictor({"inputs":preprocessed_input})
        return self.preprocess_output(preprocessed_input)



    def preprocess_input(self, query,task_id=-1):

        input_ids = [self.inputs_vocab.encode(q) for q in query]
        final_input_ids = []
        max_length = max([len(i) for i in input_ids])
        # Make them into equal shapes
        for id in input_ids:
            final_input_ids.append(id + [0] * (max_length - len(id)))

        if self.decode_hp.max_input_size > 0:
            # Subtract 1 for the EOS_ID.
            final_input_ids = final_input_ids[:self.decode_hp.max_input_size - 1]

        if self.has_input:  # Do not append EOS for pure LM tasks.
            final_id = text_encoder.EOS_ID if task_id < 0 else task_id
            final_input_ids = [i+[final_id] for i in final_input_ids]

        x = np.reshape(final_input_ids, (len(query),-1,1))

        return x



    def preprocess_output(self, translated_sentence):
            output_sentences = [self.inputs_vocab.decode(i) for i in self.estimator_decoder_predictor({"inputs":np.reshape(translated_sentence['outputs'], (len(translated_sentence['outputs']),-1,1))})['outputs']]
            return output_sentences




    def create_hparams(self):

        return trainer_lib.create_hparams(
            self.hparams_set,
            self.hparams,
            data_dir=os.path.expanduser(self.data_dir),
            problem_name=self.problem)


    def create_decode_hparams(self):
        decode_hp = decoding.decode_hparams(self.decode_hparams)
        decode_hp.shards = self.decode_shards
        decode_hp.shard_id = self.worker_id
        decode_in_memory = self.decode_in_memory or decode_hp.decode_in_memory
        decode_hp.decode_in_memory = decode_in_memory
        # decode_hp.decode_to_file = FLAGS.decode_to_file
        # decode_hp.decode_reference = FLAGS.decode_reference
        return decode_hp


    def input_fn(self):
        features = {}
        features["input_space_id"] = tf.constant(0)
        features["target_space_id"] = tf.constant(0)
        features["decode_length"] = tf.constant(58) # TODO Check with variable length_to_decode.
        features["inputs"] = tf.placeholder("int32", [None, None, 1])
        return tf.estimator.export.ServingInputReceiver(features, features)










