import torch, spaces
import gradio as gr
from diffusers import FluxPipeline

MODELS = {
    # 'FLUX.1 [dev]': 'black-forest-labs/FLUX.1-dev',
    'FLUX.1 [schnell]': 'black-forest-labs/FLUX.1-schnell',
    'OpenFLUX.1': 'ostris/OpenFLUX.1',
}
MODEL_CACHE = {}
for id, model in MODELS.items():
    print(f"Loading model {model}...")
    MODEL_CACHE[model] = FluxPipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    print(f"Loaded model {model}")

@spaces.GPU
def generate(text):
    prompt = "A cat holding a sign that says hello world"
    image = MODEL_CACHE['OpenFLUX.1'](
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    return image
    # image.save("flux-dev.png")

with gr.Blocks() as demo:
    prompt = gr.Textbox("Prompt")
    btn = gr.Button("Generate", variant="primary")
    out = gr.Image(label="Generated image", interactive=False)
    btn.click(generate,inputs=prompt,outputs=out)
demo.queue().launch()