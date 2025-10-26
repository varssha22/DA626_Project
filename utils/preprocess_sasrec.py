import collections
import numpy as np
import tensorflow as tf
from models.sasrec_model import SasRec

class SASRecPipeline:
    def __init__(self,model, vocab_size, max_context_len=15, pad_item_id=0):
        self.model = model
        self.vocab_size = vocab_size
        self.max_context_len = max_context_len
        self.pad_item_id = pad_item_id

    @staticmethod
    def get_user_sequence(df_merged, user_id):
        user_data = df_merged[df_merged["user_id"] == user_id].sort_values("order_date")
        sequence = [
            {"product_id": row.product_id, "timestamp": row.order_date, "reordered": row.reordered}
            for row in user_data.itertuples(index=False)
        ]
        return sequence

    def format_data(self, sequences):
        examples = {"sequence": [], "negative_sequence": []}
        for user_id in sequences:
            sequence = [int(d["product_id"]) for d in sequences[user_id]]

            def random_negative_item_id(low, high, positive_lst):
                sampled = np.random.randint(low=low, high=high)
                while sampled in positive_lst:
                    sampled = np.random.randint(low=low, high=high)
                return sampled

            negative_sequence = [
                random_negative_item_id(1, self.vocab_size + 1, sequence)
                for _ in range(len(sequence))
            ]

            examples["sequence"].append(np.array(sequence))
            examples["negative_sequence"].append(np.array(negative_sequence))

        examples["sequence"] = tf.ragged.constant(examples["sequence"])
        examples["negative_sequence"] = tf.ragged.constant(examples["negative_sequence"])
        return examples

    @staticmethod
    def _preprocess(example, max_context_len, pad_item_id, train=False):
        sequence = example["sequence"]
        negative_sequence = example["negative_sequence"]

        if train:
            sequence = sequence[..., :-1]
            negative_sequence = negative_sequence[..., :-1]

        batch_size = tf.shape(sequence)[0]

        if not train:
            sample_weight = tf.zeros_like(sequence, dtype="float32")[..., :-1]
            sample_weight = tf.concat(
                [sample_weight, tf.ones((batch_size, 1), dtype="float32")], axis=1
            )

        sequence = sequence.to_tensor(shape=[batch_size, max_context_len + 1], default_value=pad_item_id)
        negative_sequence = negative_sequence.to_tensor(shape=[batch_size, max_context_len + 1], default_value=pad_item_id)

        if train:
            sample_weight = tf.cast(sequence != pad_item_id, dtype="float32")
        else:
            sample_weight = sample_weight.to_tensor(shape=[batch_size, max_context_len + 1], default_value=0)

        return (
            {"item_ids": sequence[..., :-1], "padding_mask": (sequence != pad_item_id)[..., :-1]},
            {"positive_sequence": sequence[..., 1:], "negative_sequence": negative_sequence[..., 1:]},
            sample_weight[..., 1:]
        )

    @staticmethod
    def preprocess_train(examples, max_context_len, pad_item_id):
        return SASRecPipeline._preprocess(examples, max_context_len, pad_item_id, train=True)

    @staticmethod
    def preprocess_val(examples, max_context_len, pad_item_id):
        return SASRecPipeline._preprocess(examples, max_context_len, pad_item_id, train=False)

    def prepare_input(self, user_history):
        user_history = user_history[-self.max_context_len:]
        padding_needed = self.max_context_len - len(user_history)
        if padding_needed > 0:
            user_history = [self.pad_item_id] * padding_needed + user_history

        item_ids = tf.constant([user_history], dtype=tf.int32)
        padding_mask = tf.cast(item_ids == self.pad_item_id, tf.bool)
        return {"item_ids": item_ids, "padding_mask": padding_mask}

    def recommend(self, user_history, top_k=10):
        inputs = self.prepare_input(user_history)
        preds = self.model(inputs, training=False)

        if isinstance(preds, dict) and "predictions" in preds:
            scores, candidate_ids = preds["predictions"]
        elif isinstance(preds, (tuple, list)):
            scores, candidate_ids = preds
        else:
            raise ValueError(f"Unexpected model output type: {type(preds)}")

        if hasattr(candidate_ids, "numpy"):
            candidate_ids = candidate_ids.numpy()[0][:top_k]
        if hasattr(scores, "numpy"):
            scores = scores.numpy()[0][:top_k]

        return list(zip(candidate_ids, scores))
