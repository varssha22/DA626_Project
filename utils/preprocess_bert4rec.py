import numpy as np
import tensorflow as tf

class TFLiteBert4RecPipeline:
    def __init__(self, model_path, vocab_size, max_seq_len=50, pad_id=0):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.mask_token = vocab_size - 1
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    
    def prepare_input(self, user_history):
        seq = user_history[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        
        if pad_len > 0:
            seq = [self.pad_id] * pad_len + seq
        return np.array([seq], dtype=np.int32)

    
    def recommend(self, user_history, top_k=10):
        seq = self.prepare_input(user_history)[0]
        last_pos = np.max(np.where(seq != self.pad_id))
        
        masked_seq = seq.copy()
        masked_seq[last_pos] = self.mask_token
        masked_seq = np.array([masked_seq], dtype=np.int32)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], masked_seq)
        self.interpreter.invoke()

        logits = self.interpreter.get_tensor(self.output_details[0]['index'])
        scores = logits[0, last_pos]  

        scores[self.pad_id] = -1e9
        scores[self.mask_token] = -1e9
        top_items = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_items]

        return list(zip(top_items, top_scores))
