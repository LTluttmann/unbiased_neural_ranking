import tensorflow.keras as nn
from mt.models.layers import create_tower


class PBKClassifier(nn.Model):
    def __init__(
            self, 
            encoder, 
            vocab, 
            hidden_layers=[512, 512],
            dropout_rate=0.5,
            activation="relu",
            input_batch_norm = True,
            use_batch_norm = True,
            finetune=False
        ):
        super().__init__()
        self.encoder = encoder
        if not finetune:
            self.encoder.trainable=False
        self.dropout = nn.layers.Dropout(dropout_rate)
        self.mlp = create_tower(
            hidden_layer_dims=hidden_layers, 
            output_units=len(vocab)+1, 
            activation=activation, 
            output_activation="softmax", 
            input_batch_norm=input_batch_norm,
            use_batch_norm=use_batch_norm, 
            dropout=dropout_rate)

    def call(self, query, training=True):
        x = self.encoder(query, training=training)
        x = self.dropout(x, training=training)
        out = self.mlp(x, training=training)
        return out