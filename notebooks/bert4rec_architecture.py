import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import collections
from tensorflow.keras.preprocessing.sequence import pad_sequences

def masked_sparse_categorical_crossentropy(y_true, y_pred):
    mask = tf.not_equal(y_true, -100)             
    mask_f = tf.cast(mask, tf.float32)

    y_true_safe = tf.where(mask, y_true, tf.zeros_like(y_true))

    per_token_loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true_safe, y_pred, from_logits=True
    ) 

    per_token_loss = per_token_loss * mask_f
    denom = tf.reduce_sum(mask_f) + 1e-8
    return tf.reduce_sum(per_token_loss) / denom

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        warmup_lr = self.initial_learning_rate * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
        return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: self.decay_schedule_fn(step - self.warmup_steps))

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_schedule_fn": None
        }

def transformer_encoder(x, num_heads, ff_dim, dropout, emb_dim):
    key_dim = max(emb_dim // num_heads, 1)
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = layers.Dropout(dropout)(attn)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ffn = layers.Dense(ff_dim, activation='gelu')(out1)
    ffn = layers.Dense(emb_dim)(ffn)
    ffn = layers.Dropout(dropout)(ffn)

    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)


def build_bert4rec(vocab_size, max_len, emb_dim=128, num_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32, name='input_ids')

    token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True, name='token_embedding')(inputs)

    positions = tf.range(start=0, limit=max_len, delta=1)
    pos_layer = layers.Embedding(input_dim=max_len, output_dim=emb_dim, name='pos_embedding')
    pos_emb = pos_layer(positions)
    pos_emb = tf.expand_dims(pos_emb, axis=0) 
    x = token_emb + pos_emb                   

    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout, emb_dim=emb_dim)

    logits = layers.Dense(vocab_size, name='logits')(x)

    model = models.Model(inputs=inputs, outputs=logits)
    initial_lr = 1e-3 
    total_steps = 10000
    warmup_steps = int(0.1 * total_steps)
    
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps - warmup_steps,
        end_learning_rate=0.0
    )
    schedule = WarmUp(initial_lr, lr_schedule, warmup_steps)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=2.0
    )
    model.compile(optimizer=optimizer, loss=masked_sparse_categorical_crossentropy)
    return model
