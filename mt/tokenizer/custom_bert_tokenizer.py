import tensorflow as tf
import tensorflow.keras as nn
import tensorflow_text as text

import pathlib
import re


class CustomBertTokenizer(tf.Module):
    """just a serializable version of BertTokenizer. Can be saved"""
    def __init__(self, vocab_path):
        self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
        vocab = pathlib.Path(vocab_path).read_text().splitlines()

        self._reserved_tokens = ["[PAD]", "[UNK]", "[START]", "[END]"]
        self._vocab_path = tf.saved_model.Asset(vocab_path)

        self.vocab = tf.Variable(vocab, trainable=False)

        # Create the signatures for export:

        # Include a tokenize signature for a batch of strings.
        self.tokenize.get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string))

        # Include `detokenize` and `lookup` signatures for:
        #   * `Tensors` with shapes [tokens] and [batch, tokens]
        #   * `RaggedTensors` with shape [batch, tokens]
        self.detokenize.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.detokenize.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.lookup.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.int64))
        self.lookup.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

        self.cleanup_text.get_concrete_function(
            tf.TensorSpec(shape=[None, None], dtype=tf.string))
        self.cleanup_text.get_concrete_function(
            tf.RaggedTensorSpec(shape=[None, None], dtype=tf.string))

        self.get_vocab_size.get_concrete_function()
        self.get_vocab_path.get_concrete_function()
        self.get_reserved_tokens.get_concrete_function()

    @tf.function
    def tokenize(self, strings):
        enc = self.tokenizer.tokenize(strings)
        # Merge the `word` and `word-piece` axes.
        # enc = enc.merge_dims(-2, -1).merge_dims(-2, -1)
        return enc

    @tf.function
    def detokenize(self, tokenized):
        words = self.tokenizer.detokenize(tokenized)
        return self.cleanup_text(words)

    @tf.function
    def lookup(self, token_ids):
        return tf.gather(self.vocab, token_ids)

    @tf.function
    def get_vocab_size(self):
        return tf.shape(self.vocab)[0]

    @tf.function
    def get_vocab_path(self):
        return self._vocab_path

    @tf.function
    def get_reserved_tokens(self):
        return tf.constant(self._reserved_tokens)

    @tf.function
    def cleanup_text(self, token_txt):
        # Drop the reserved tokens, except for "[UNK]".
        bad_tokens = [re.escape(tok)
                      for tok in self._reserved_tokens if tok != "[UNK]"]
        bad_token_re = "|".join(bad_tokens)

        bad_cells = tf.strings.regex_full_match(token_txt, bad_token_re)
        result = tf.ragged.boolean_mask(token_txt, ~bad_cells)

        # Join them into strings.
        result = tf.strings.reduce_join(result, separator=' ', axis=-1)
        return result


class BertTokenizerLayer(nn.layers.Layer):
    def __init__(self, vocab_file, **kwargs):
        super().__init__(**kwargs)
        self.vocab_file = vocab_file
        self.tokenizer = CustomBertTokenizer(self.vocab_file)
        self.tokenize = nn.layers.Lambda(lambda text_input: self.tokenizer.tokenize(text_input), name="tokenizer")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def call(self, input):
        tokens = self.tokenize(input)
        flattened_tokens = tokens.merge_dims(-2,-1).merge_dims(-2,-1)
        return flattened_tokens

    def get_config(self):
        return {
            "vocab_file": self.vocab_file,
            **super().get_config()
        }