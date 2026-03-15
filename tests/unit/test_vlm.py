from types import SimpleNamespace

from prime_rl.utils.vlm import get_layer_prefix, is_vlm_config, is_vlm_model


def test_ministral3_name_is_detected_as_vlm():
    assert is_vlm_model("mistralai/Ministral-3-14B-Instruct-2512-BF16")


def test_mistral3_model_type_is_detected_as_vlm():
    config = SimpleNamespace(model_type="mistral3")
    assert is_vlm_config(config)


def test_mistral3_uses_language_model_layer_prefix():
    config = SimpleNamespace(model_type="mistral3")
    assert get_layer_prefix(config) == "model.language_model.layers."
