import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
import collections
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MaskedItemModeling(layers.Layer):
    def __init__(self, mask_token_id, mask_prob=0.15):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.mask_prob = mask_prob

    def call(self, inputs, training=None):
        if not training:
            return inputs, tf.zeros_like(inputs, dtype=tf.bool)
        # Random mask positions
        mask = tf.random.uniform(shape=tf.shape(inputs)) < self.mask_prob
        masked_inputs = tf.where(mask, self.mask_token_id, inputs)
        return masked_inputs, mask

def transformer_encoder(inputs, num_heads, ff_dim, dropout):
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=inputs.shape[-1]
    )(inputs, inputs)

    attn_output = layers.Dropout(dropout)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn_output = layers.Dense(ff_dim, activation='relu',
                              kernel_regularizer=regularizers.l2(0.01))(out1)
    ffn_output = layers.Dense(inputs.shape[-1],
                              kernel_regularizer=regularizers.l2(0.01))(ffn_output)
    ffn_output = layers.Dropout(dropout)(ffn_output)

    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_bert4rec(
    vocab_size,
    max_len,
    emb_dim=128,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout=0.2,
    initial_lr=1e-1,
    end_lr=1e-4,
    decay_steps=10000,
    mask_prob=0.2,
):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)

    mask_layer = MaskedItemModeling(mask_token_id=vocab_size - 1, mask_prob=mask_prob)
    masked_inputs, mask_positions = mask_layer(inputs)

    token_emb = layers.Embedding(
        vocab_size,
        emb_dim,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
        embeddings_regularizer=regularizers.l2(0.01)
    )(masked_inputs)

    pos_emb = layers.Embedding(
        max_len,
        emb_dim,
        embeddings_initializer=tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
    )(tf.range(start=0, limit=max_len))

    x = token_emb + pos_emb
    x = layers.Dropout(dropout)(x)


    for _ in range(num_layers):
        x = transformer_encoder(x, num_heads, ff_dim, dropout)


    outputs = layers.Dense(vocab_size, activation='softmax',
                           kernel_regularizer=regularizers.l2(0.01))(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=1.0 
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        clipnorm=5.0 
    )
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    return model
