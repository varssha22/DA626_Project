import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from bert4rec_architecture import build_bert4rec

DATA_DIR = "data/"

with open(os.path.join(DATA_DIR, "train_sequences.pkl"), "rb") as f:
    train_sequences = pickle.load(f)

with open(os.path.join(DATA_DIR, "val_sequences.pkl"), "rb") as f:
    val_sequences = pickle.load(f)

with open(os.path.join(DATA_DIR, "test_sequences.pkl"), "rb") as f:
    test_sequences = pickle.load(f)


MAX_SEQ_LEN = 50
VOCAB_SIZE = 49688
MASK_TOKEN = VOCAB_SIZE - 1

def pad_seq_data(sequences):
    return pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre')

X_train = pad_seq_data(train_sequences)
X_val = pad_seq_data(val_sequences)
X_test = pad_seq_data(test_sequences)

print("X_train shape:", X_train.shape)
print("Sample values:", X_train[0][:10])
print("Min:", X_train.min(), "Max:", X_train.max())

def prepare_masked_inputs(sequences, mask_token_id, vocab_size, mask_prob=0.15, seed=None):
    rng = np.random.RandomState(seed)
    masked_inputs, labels = [], []
    for seq in sequences:
        seq = seq.copy()
        label = np.full_like(seq, fill_value=-100, dtype=np.int32)

        for i in range(len(seq)):
            if seq[i] == 0:
                continue

            if rng.rand() < mask_prob:
                label[i] = seq[i]
                r = rng.rand()

                if r < 0.8:
                    seq[i] = mask_token_id
                elif r < 0.9:
                    seq[i] = rng.randint(1, vocab_size - 1)
                else:
                    pass

        masked_inputs.append(seq)
        labels.append(label)

    return np.array(masked_inputs, dtype=np.int32), np.array(labels, dtype=np.int32)

X_masked_train, y_masked_train = prepare_masked_inputs(X_train, MASK_TOKEN, VOCAB_SIZE)
X_masked_val,   y_masked_val   = prepare_masked_inputs(X_val, MASK_TOKEN, VOCAB_SIZE)

model = build_bert4rec(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_SEQ_LEN,
    emb_dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    dropout=0.1
)

model.summary()
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "checkpoints/bert4rec_best.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min"
)

nan_cb = tf.keras.callbacks.TerminateOnNaN()

batch_x = X_masked_train[:32]
batch_y = y_masked_train[:32]
loss = model.train_on_batch(batch_x, batch_y)
print("One-batch loss:", loss)

history = model.fit(
    X_masked_train, y_masked_train,
    validation_data=(X_masked_val, y_masked_val),
    epochs=10,
    batch_size=128,
    callbacks=[checkpoint_cb, nan_cb],
    verbose=1
)

