import multiprocessing as mp
import torch
import os
from functools import partial
import gradio as gr
import traceback
from huggingface_hub import hf_hub_download, snapshot_download
from tts.infer_cli import MegaTTS3DiTInfer, convert_to_wav, cut_wav


def download_weights():
    """Download model weights from HuggingFace if not already present."""
    repo_id = "mrfakename/MegaTTS3-VoiceCloning"
    weights_dir = "weights"
    
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


def model_worker(input_queue, output_queue, device_id):
    device = None
    if device_id is not None:
        device = torch.device(f'cuda:{device_id}')
    infer_pipe = MegaTTS3DiTInfer(device=device)

    while True:
        task = input_queue.get()
        inp_audio_path, inp_text, infer_timestep, p_w, t_w = task
        try:
            convert_to_wav(inp_audio_path)
            wav_path = os.path.splitext(inp_audio_path)[0] + '.wav'
            cut_wav(wav_path, max_len=28)
            with open(wav_path, 'rb') as file:
                file_content = file.read()
            resource_context = infer_pipe.preprocess(file_content)
            wav_bytes = infer_pipe.forward(resource_context, inp_text, time_step=infer_timestep, p_w=p_w, t_w=t_w)
            output_queue.put(wav_bytes)
        except Exception as e:
            traceback.print_exc()
            print(task, str(e))
            output_queue.put(None)


def generate_speech(inp_audio, inp_text, infer_timestep, p_w, t_w, processes, input_queue, output_queue):
    if not inp_audio or not inp_text:
        gr.Warning("Please provide both reference audio and text to generate.")
        return None
    
    print("Generating speech with:", inp_audio, inp_text, infer_timestep, p_w, t_w)
    input_queue.put((inp_audio, inp_text, infer_timestep, p_w, t_w))
    res = output_queue.get()
    if res is not None:
        return res
    else:
        gr.Warning("Speech generation failed. Please try again.")
        return None


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    mp_manager = mp.Manager()

    devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if devices != '':
        devices = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(",")
    else:
        devices = None
    
    num_workers = 1
    input_queue = mp_manager.Queue()
    output_queue = mp_manager.Queue()
    processes = []

    print("Starting workers...")
    for i in range(num_workers):
        p = mp.Process(target=model_worker, args=(input_queue, output_queue, i % len(devices) if devices is not None else None))
        p.start()
        processes.append(p)

    with gr.Blocks(title="MegaTTS3 Voice Cloning") as demo:
        gr.Markdown("# MegaTTS3 Voice Cloning")
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
            fn=partial(generate_speech, processes=processes, input_queue=input_queue, output_queue=output_queue),
            inputs=[reference_audio, text_input, infer_timestep, p_w, t_w],
            outputs=[output_audio]
        )

    demo.launch(server_name='0.0.0.0', server_port=7860, debug=True)
    
    for p in processes:
        p.join()