import spaces
import torch
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import gradio as gr
import traceback
from huggingface_hub import snapshot_download
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def download_weights():
    """Download model weights from HuggingFace if not already present."""
    repo_id = "mrfakename/MegaTTS3-VoiceCloning"
    weights_dir = "checkpoints"
    
    if not os.path.exists(weights_dir):
        print("Downloading model weights from HuggingFace...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=weights_dir,
            local_dir_use_symlinks=False
        )
        print("Model weights downloaded successfully!")
    else:
        print("Model weights already exist.")
    
    return weights_dir


# Download weights and initialize model
download_weights()
print("Initializing MegaTTS3 model...")
infer_pipe = MegaTTS3DiTInfer()
print("Model loaded successfully!")

@spaces.GPU
def generate_speech(inp_audio, inp_text, infer_timestep, p_w, t_w):
    if not inp_audio or not inp_text:
        gr.Warning("Please provide both reference audio and text to generate.")
        return None
    
    try:
        print(f"Generating speech with: {inp_text}...")
        
        # Convert and prepare audio
        convert_to_wav(inp_audio)
        wav_path = os.path.splitext(inp_audio)[0] + '.wav'
        cut_wav(wav_path, max_len=28)
        
        # Read audio file
        with open(wav_path, 'rb') as file:
            file_content = file.read()
        
        # Generate speech
        resource_context = infer_pipe.preprocess(file_content)
        wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
        
        return wav_bytes
    except Exception as e:
        traceback.print_exc()
        gr.Warning(f"Speech generation failed: {str(e)}")
        return None


with gr.Blocks(title="MegaTTS3 Voice Cloning") as demo:
    gr.Markdown("# MegaTTS 3 Voice Cloning")
    gr.Markdown("MegaTTS 3 is a text-to-speech model trained by ByteDance with exceptional voice cloning capabilities. The original authors did not release the WavVAE encoder, so voice cloning was not publicly available; however, thanks to [@ACoderPassBy](https://modelscope.cn/models/ACoderPassBy/MegaTTS-SFT)'s WavVAE encoder, we can now clone voices with MegaTTS 3!")
    gr.Markdown("This is by no means the best voice cloning solution, but it works pretty well for some specific use-cases. Try out multiple and see which one works best for you.")
    gr.Markdown("**Please use this Space responsibly and do not abuse it!**")
    gr.Markdown("h/t to MysteryShack on Discord for the info about the unofficial WavVAE encoder!")
    gr.Markdown("Upload a reference audio clip and enter text to generate speech with the cloned voice.")
    
    with gr.Row():
        with gr.Column():
            reference_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                sources=["upload", "microphone"]
            )
            text_input = gr.Textbox(
                label="Text to Generate",
                placeholder="Enter the text you want to synthesize...",
                lines=3
            )
            
            with gr.Accordion("Advanced Options", open=False):
                infer_timestep = gr.Number(
                    label="Inference Timesteps",
                    value=32,
                    minimum=1,
                    maximum=100,
                    step=1
                )
                p_w = gr.Number(
                    label="Intelligibility Weight",
                    value=1.4,
                    minimum=0.1,
                    maximum=5.0,
                    step=0.1
                )
                t_w = gr.Number(
                    label="Similarity Weight", 
                    value=3.0,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Audio")
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
        outputs=[output_audio]
    )

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True)