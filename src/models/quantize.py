from transformers import BitsAndBytesConfig


quantization_4bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

quantization_8bit_config = BitsAndBytesConfig(
    load_in_8bit=True
)
