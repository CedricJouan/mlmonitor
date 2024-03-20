# SPDX-License-Identifier: Apache-2.0
from tensorflow.keras.layers import TextVectorization, Embedding, GRU, Dense
from tensorflow.keras.models import Model


class TextSummarizer(Model):
    def __init__(self, vocab_size, embedding_dim=256, gru_units=128):
        super(TextSummarizer, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.encoder = GRU(gru_units, return_sequences=True, return_state=True)
        self.decoder = GRU(gru_units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)

    def call(self, inputs, training=False):
        text, summary = inputs
        text_embedded = self.embedding(text)
        _, state = self.encoder(text_embedded)
        
        summary_embedded = self.embedding(summary)
        decoder_output, _ = self.decoder(summary_embedded, initial_state=state)
        
        output = self.dense(decoder_output)
        return output

def base_model(vocab_size) -> Model:
    return TextSummarizer(vocab_size=vocab_size)

