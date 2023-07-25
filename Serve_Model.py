# supress all cuda warning
import warnings
warnings.filterwarnings("ignore")
import socket
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, \
    GenerationConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from typing import Callable


class ServeModel:
    def __init__(self, port: int = 8080, model_name: str = 'openlm-research/open_llama_7b_v2',
                 model_path: str = "./model2/") -> None:
        """
        Will initialize the class
        """
        self.port = port
        self.model_name = model_name
        self.model_path = model_path
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('', self.port))
        self.socket.listen(5)
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self) -> Callable:
        """
        Will load the model
        :return: Automodel
        """
        peft_model_base = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16).to('cuda')
        return PeftModel.from_pretrained(peft_model_base, self.model_path, torch_dtype=torch.bfloat16,
                                         is_trainable=False)

    def load_tokenizer(self) -> Callable:
        """
        Will load the tokenizer
        :return: AutoTokenizer
        """
        return AutoTokenizer.from_pretrained(self.model_name)

    def serve(self) -> None:
        """
        Will serve the model In / Output
        :return: None
        """
        print("Serving Model - awaiting connection")
        while True:
            conn, addr = self.socket.accept()
            print("Connected to", addr)
            # receive the data - 1024 bytes a string
            data = conn.recv(1024).decode()
            if not data:
                break
            prompt = f"""
            Extract the Brithdate in the text and output it in ISO8601 Format: {data}

            Output:
            """
            input_ids = self.tokenizer(prompt, return_tensors="pt").to('cuda').input_ids
            output = self.model.generate(input_ids=input_ids,
                                         generation_config=GenerationConfig(max_new_tokens=11,
                                                                            num_beams=1),
                                         pad_token_id=self.tokenizer.eos_token_id)
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            # only keep everything after the word Output:
            text = text[text.find("Output:") + len("Output:"):]
            # delete all leading and trailing whitespaces
            text = text.strip()
            # send the text back
            conn.sendall(text.encode())
            conn.close()
            print("Connection closed")


if __name__ == '__main__':
    model = ServeModel()
    model.serve()
