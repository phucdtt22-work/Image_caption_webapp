from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
from typing import List, Union
from tqdm import tqdm
import torch
import torch.amp.autocast_mode
import torch.nn as nn
import os



# Constants
VLM_PROMPT = "A descriptive caption for this image:\n"
CLIP_PATH = "google/siglip-so400m-patch14-384"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
CHECKPOINT_PATH = Path("wpkklhc6")
UPLOAD_FOLDER = 'static'



# Define Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define the ImageAdapter class and load_models function
class ImageAdapter(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_features, output_features)
        self.activation = torch.nn.GELU()
        self.linear2 = torch.nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        return self.linear2(self.activation(self.linear1(vision_outputs)))

def load_models():
    print("Loading CLIP 📎")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model.eval().requires_grad_(False).to("cuda")

    print("Loading tokenizer 🪙")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    print("Loading LLM 🤖")
    text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="cuda", torch_dtype=torch.bfloat16).eval()

    print("Loading image adapter 🖼️")
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load(CHECKPOINT_PATH / "image_adapter.pt", map_location="cuda", weights_only=False))
    image_adapter.eval().to("cuda")

    return clip_processor, clip_model, tokenizer, text_model, image_adapter

@torch.no_grad()
def stream_chat(input_images: List[Image.Image], batch_size: int, pbar: tqdm, models: tuple, prompt: str) -> List[str]:
    clip_processor, clip_model, tokenizer, text_model, image_adapter = models
    device = 'cuda'
    torch.cuda.empty_cache()
    all_captions = []


    # Move models to the correct device
    clip_model.to(device)
    image_adapter.to(device)


    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i+batch_size]
        
        try:
            images = clip_processor(images=batch, return_tensors='pt', padding=True).pixel_values.to(device)
        except ValueError as e:
            print(f"Error processing image batch: {e}")
            print("Skipping this batch and continuing...")
            continue

        # Ensure both model and data are in the same dtype
        with torch.amp.autocast('cuda',enabled=False):  # Disable autocast if not needed
            vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features).to(dtype=torch.bfloat16, device=device)

        prompt_encoded = tokenizer.encode(prompt, return_tensors='pt').to(device)
        prompt_embeds = text_model.get_input_embeddings()(prompt_encoded).to(dtype=torch.bfloat16, device=device)
        embedded_bos = text_model.get_input_embeddings()(torch.tensor([[tokenizer.bos_token_id]], device=device, dtype=torch.int64)).to(dtype=torch.bfloat16, device=device)

        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images,
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1).to(dtype=torch.bfloat16, device=device)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1).to("cuda"),
            torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long).to("cuda"),
            prompt_encoded.expand(embedded_images.shape[0], -1).to("cuda"),
        ], dim=1).to(device=device)

        attention_mask = torch.ones_like(input_ids).to(device)


        generate_ids = text_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            top_k=10,
            temperature=0.5,
        )

        generate_ids = generate_ids[:, input_ids.shape[1]:]

        for ids in generate_ids:
            caption = tokenizer.decode(ids[:-1] if ids[-1] == tokenizer.eos_token_id else ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            caption = caption.replace('<|end_of_text|>', '').replace('<|finetune_right_pad_id|>', '').strip()
            all_captions.append(caption)

        if pbar:
            pbar.update(len(batch))

    return all_captions




# Load the models once
models = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = 'uploaded_image.jpg'  # You can use a unique filename to avoid conflicts
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Generate caption
        image = Image.open(file_path).convert('RGB')
        prompt = VLM_PROMPT
        captions = stream_chat([image], 1, None, models, prompt)

        return render_template('result.html', caption=captions[0])
    else:
        return "Invalid file type, only images are allowed."

if __name__ == '__main__':
    app.run(debug = True)
