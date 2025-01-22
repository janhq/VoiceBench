from .base import VoiceAssistant
import torch
import os
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torchaudio
from ichigo_whisper.config.vq_config import VQConfig
from ichigo_whisper.models.factory import make_vq_model
def load_vq_model(
        ref,
        size: str,
        repo_id=None,
        filename=None,
        local_dir=None,
        local_filename=None,
    ):
        """Load model from file or Hugging Face Hub.

        Args:
            ref (str): Either a local path or "repo_id:filename" format
            repo_id (str, optional): Hugging Face repository ID
            filename (str, optional): Filename in the repository
            local_dir (str, optional): Local directory for downloads
            local_filename (str, optional): Direct path to local file

        Returns:
            RQBottleneckTransformer: Loaded model instance
        """
        # Parse reference string
        if repo_id is None and filename is None and local_filename is None:
            if ":" in ref:
                repo_id, filename = ref.split(":", 1)
            else:
                local_filename = ref

        # Download or use local file
        if not os.path.exists(f"{local_filename}"):
            local_filename = hf_hub_download(
                repo_id=repo_id, filename=filename, local_dir=local_dir
            )

        # Load and validate spec
        spec = torch.load(local_filename)
        model_state_dict = {
            k.replace("model.", ""): v for k, v in spec["state_dict"].items()
        }
        vq_config = VQConfig()
        ichigo_model = make_vq_model(size=size, config=vq_config)
        ichigo_model.load_state_dict(model_state_dict)
        ichigo_model.eval()
        return ichigo_model
    
def convert_ids_to_tokens(id_list):
    """
    Convert a list of IDs to a compressed sound token string.
    
    Args:
        id_list (list): List of sound IDs
    
    Returns:
        str: Formatted string with sound tokens and duration
    """
    if not id_list:
        return "<|sound_start|><|sound_end|>"
    
    result = ["<|sound_start|>"]
    i = 0
    
    while i < len(id_list):
        current_id = id_list[i]
        count = 1
        
        # Count consecutive occurrences of the same ID
        while i + count < len(id_list) and id_list[i + count] == current_id:
            count += 1
            
        # Add duration token if count > 1
        if count > 1:
            result.append(f"<|duration_{str(count).zfill(2)}|>")
            
        # Add the sound token (each ID separately)
        result.append(f"<|sound_{str(current_id).zfill(4)}|>")
        
        # Move index forward
        i += count
    
    result.append("<|sound_end|>")
    return "".join(result)

class Ichigov05Assistant(VoiceAssistant):
    def __init__(self):
        self.device = "cuda"
        ichigo_name = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth"
        model_size = "merge-medium-vi-2d-2560c-dim64"
        self.ichigo_model = load_vq_model(ichigo_name, model_size)
        self.ichigo_model.ensure_whisper(self.device)
        self.ichigo_model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained('/home/root/BachVD/model_zoo/Ichigo-llama3.1-8B-v0.5', cache_dir='./cache')
        self.model = AutoModelForCausalLM.from_pretrained('/home/root/BachVD/model_zoo/Ichigo-llama3.1-8B-v0.5', device_map='cuda', torch_dtype=torch.bfloat16, cache_dir='./cache')
        self.prompt_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    def audio_to_sound_tokens(self, wav, sr, device='cuda'):  
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            codes = self.ichigo_model.quantize(wav.to(device))
            codes = codes[0].cpu().tolist()
        
        result = convert_ids_to_tokens(codes)
        return f'{result}'
    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        sound_tokens = self.audio_to_sound_tokens(torch.from_numpy(audio['array']).float().unsqueeze(0), audio['sampling_rate'])
        message = self.prompt_template.format(text=sound_tokens)
        input_ids = self.tokenizer([message], return_tensors='pt').to(self.device)

        generated_ids = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids.input_ids, generated_ids)
        ]
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

