from langchain_huggingface import HuggingFacePipeline


class Phi3Pipe:
    def __init__(self):
        self.phi_3_pipe = HuggingFacePipeline.from_model_id(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            task="text-generation",
            device=0,
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "do_sample": True,
                "repetition_penalty": 1.03,
                "top_k": 50,
                "temperature": 0.55,
                "return_full_text": False
            },
            # model_kwargs={
            #     "quantization_config": quantization_8bit_config
            # }
        )

    def get_pipe(self):
        return self.phi_3_pipe
