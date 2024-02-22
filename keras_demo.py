import keras
import keras_nlp
import numpy as np

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
gemma_lm.generate("Keras is a", max_length=30)

# Generate with batched prompts.
gemma_lm.generate(["Keras is a", "I want to say"], max_length=30)

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
gemma_lm.compile(sampler="top_k")
gemma_lm.generate("I want to say", max_length=30)

gemma_lm.compile(sampler=keras_nlp.samplers.BeamSampler(num_beams=2))
gemma_lm.generate("I want to say", max_length=30)

prompt = {
    # `2, 214064, 603` maps to the start token followed by "Keras is".
    "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
    # Use `"padding_mask"` to indicate values that should not be overridden.
    "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
}

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
    "gemma_instruct_7b_en",
    preprocessor=None,
)
gemma_lm.generate(prompt)

features = ["The quick brown fox jumped.", "I forgot my homework."]
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_instruct_7b_en")
gemma_lm.fit(x=features, batch_size=2)

x = {
    "token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 3, 0]] * 2),
    "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
}
y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)

gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset(
    "gemma_instruct_7b_en",
    preprocessor=None,
)
gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)