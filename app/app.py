import gradio as gr, torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline

pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.load_lora_weights("equipe/lora-estilo-cordel")
pipe = pipe.to("cuda")

expansor = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

tts = pipeline("text-to-speech", model="suno/bark-small")

def gerar(tema):
 instrucao = (f"Expanda o tema '{tema}' em uma descricao visual rica, em uma frase, iniciando com 'estilo_cordel,'")
 prompt = expansor(instrucao, max_new_tokens=60)[0]["generated_text"]
 imagem = pipe(prompt, negative_prompt="desfocado, deformado", guidance_scale=7.5, num_inference_steps=30).images[0]
 audio = tts(prompt)
 return prompt, imagem, (audio["sampling_rate"], audio["audio"])

demo = gr.Interface(gerar, gr.Textbox(label="Tema"),
[gr.Textbox(label="Prompt expandido"), gr.Image(), gr.Audio()],
 title="Ateliê Generativo — Equipe X")

demo.launch()
