# Copyright (c) 2024, please contact before redistribution/modification

FONT_URL = 'https://fonts.gstatic.com/s/inter/v13/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hjQ.ttf'
from moviepy.editor import *
import whisper
from cached_path import cached_path
from moviepy.video.tools.subtitles import SubtitlesClip
import spaces
import torch
import tempfile
import gradio as gr
mdl = whisper.load_model("base")
if torch.cuda.is_available(): mdl.to('cuda')
@spaces.GPU(enable_queue=True)
def subtitle(input):
    status = "**Starting...**"
    yield status, gr.update()
    gr.Info("Transcribing...")
    status += "\n\n[1/5] Transcribing... (may take a while)"
    yield status, gr.update()
    transcript = mdl.transcribe(
        word_timestamps=True,
        audio=input
    )
    status += "\n\n[2/5] Processing subtitles..."
    yield status, gr.update()
    gr.Info("Processing subtitles...")
    subs = []
    for segment in transcript['segments']:
        for word in segment['words']:
            subs.append(((word['start'], word['end'],), word['word'].strip(),))
    status += "\n\n[3/5] Loading video..."
    yield status, gr.update()
    gr.Info("Loading video...")
    video = VideoFileClip(input)
    width, height = video.size
    gr.Info(width)
    generator = lambda txt: TextClip(txt, size=(width * (3 / 4) + 8, None), color='white', stroke_color='black', stroke_width=8, method='caption', fontsize=min(width / 7, height / 7), font=str(cached_path(FONT_URL)))
    generator1 = lambda txt: TextClip(txt, size=(width * (3 / 4), None), color='white', method='caption', fontsize=min(width / 7, height / 7), font=str(cached_path(FONT_URL)))
    status += "\n\n[4/5] Loading video clip..."
    yield status, gr.update()
    gr.Info("Loading video clip...")
    subtitles = SubtitlesClip(subs, generator)
    subtitles2 = SubtitlesClip(subs, generator1)
    result_1 = CompositeVideoClip([video, subtitles.set_pos(('center','center'))])
    result = CompositeVideoClip([result_1, subtitles2.set_pos(('center','center'))])
    status += "\n\n[5/5] Writing video... (may take a while)"
    yield status, gr.update()
    gr.Info("Writing video...")
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        result.write_videofile(f.name, codec='h264_videotoolbox', audio_codec='aac', threads=64)
    status += "\n\n**Done!**"
    yield status, f.name
    return

with gr.Blocks() as demo:
    gr.Markdown("""
# AutoSubs

Automatically add on-screen subtitles to your videos.

**NOTE:** Uploading copyrighted/NSFW content to this service is strictly prohibited.

The maximum length of video is 15 minutes. This service probably won't work well on non-English videos.

Powered by OAI Whisper & MoviePy!
""")
    status = gr.Markdown("**Status updates will appear here.**")
    vid_inp = gr.Video(interactive=True, label="Upload or record video", max_length=900)
    go_btn = gr.Button("Transcribe!", variant="primary")
    vid_out = gr.Video(interactive=False, label="Result")
    go_btn.click(subtitle, inputs=[vid_inp], outputs=[status, vid_out])
demo.queue(api_open=False).launch(show_api=False)