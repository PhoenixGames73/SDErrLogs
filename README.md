# SDErrLogs
Error logs from SD

venv "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\Scripts\Python.exe"
fatal: No names found, cannot describe anything.
Python 3.10.6 (tags/v3.10.6:9c7b4bd, Aug  1 2022, 21:53:49) [MSC v.1932 64 bit (AMD64)]
Version: ## 1.4.1
Commit hash: 9d53f44a88adea631aa4c7ab549ba8d1a620fa92
Installing requirements
Warning: caught exception 'Torch not compiled with CUDA enabled', memory monitor disabled
If submitting an issue on github, please provide the full startup log for debugging purposes.

Initializing Dreambooth
Dreambooth revision: c2a5617c587b812b5a408143ddfb18fc49234edf
Successfully installed accelerate-0.19.0 fastapi-0.94.1 gitpython-3.1.32 transformers-4.30.2


Does your project take forever to startup?
Repetitive dependency installation may be the reason.
Automatic1111's base project sets strict requirements on outdated dependencies.
If an extension is using a newer version, the dependency is uninstalled and reinstalled twice every startup.

[+] xformers version 0.0.20 installed.
[+] torch version 2.0.0 installed.
[+] torchvision version 0.15.1 installed.
[+] accelerate version 0.19.0 installed.
[+] diffusers version 0.16.1 installed.
[+] transformers version 4.30.2 installed.
[+] bitsandbytes version 0.35.4 installed.

Launching Web UI with arguments: --autolaunch --lowvram --opt-sub-quad-attention --disable-nan-check
C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\pkg_resources\__init__.py:123: PkgResourcesDeprecationWarning: ansformers is an invalid version and will not be supported in a future release
  warnings.warn(
No module 'xformers'. Proceeding without it.
Warning: caught exception '', memory monitor disabled
Loading weights [fbd8419662] from C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\models\Stable-diffusion\ymix_31.safetensors
Creating model from config: C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\configs\v1-inference.yaml
LatentDiffusion: Running in eps-prediction mode
DiffusionWrapper has 859.52 M params.
Textual inversion embeddings loaded(1): Bollard penetration
Model loaded in 63.1s (load weights from disk: 2.5s, create model: 0.7s, apply weights to model: 58.5s, apply half(): 1.0s, calculate empty prompt: 0.2s).
Applying attention optimization: sub-quadratic... done.
CUDA SETUP: Loading binary C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\bitsandbytes\libbitsandbytes_cudaall.dll...
preload_extensions_git_metadata for 8 extensions took 0.30s
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
Startup time: 82.1s (import torch: 3.9s, import gradio: 1.8s, import ldm: 0.9s, other imports: 2.6s, list SD models: 0.2s, load scripts: 68.4s, create ui: 3.1s, gradio launch: 1.0s).
Custom model name is AutoF
Initializing dreambooth training...
Pre-processing images: Class: : 62it [00:00, 79.79it/s]
We need a total of 31 class images.                                                     | 1/31 [00:00<00:23,  1.29it/s]
Generating 31 class images for training...
                                                                                                                       Traceback (most recent call last):0%|                                                            | 0/31 [00:00<?, ?it/s]
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\dreambooth\ui_functions.py", line 729, in start_training
    result = main(class_gen_method=class_gen_method)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\dreambooth\train_dreambooth.py", line 1548, in main
    return inner_loop()
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\dreambooth\memory.py", line 119, in decorator
    return function(batch_size, grad_size, prof, *args, **kwargs)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\dreambooth\train_dreambooth.py", line 246, in inner_loop
    count, instance_prompts, class_prompts = generate_classifiers(
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\dreambooth\utils\gen_utils.py", line 153, in generate_classifiers
    builder = ImageBuilder(
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\extensions\sd_dreambooth_extension\helpers\image_builder.py", line 88, in __init__
    self.image_pipe = DiffusionPipeline.from_pretrained(config.get_pretrained_model_name_or_path(), torch_dtype=torch.float16)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\diffusers\pipelines\pipeline_utils.py", line 1039, in from_pretrained
    loaded_sub_model = load_sub_model(
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\diffusers\pipelines\pipeline_utils.py", line 445, in load_sub_model
    loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\modeling_utils.py", line 2675, in from_pretrained
    model = cls(config, *model_args, **model_kwargs)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 782, in __init__
    self.text_model = CLIPTextTransformer(config)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 700, in __init__
    self.encoder = CLIPEncoder(config)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 585, in __init__
    self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 585, in <listcomp>
    self.layers = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 358, in __init__
    self.self_attn = CLIPAttention(config)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\transformers\models\clip\modeling_clip.py", line 252, in __init__
    self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\torch\nn\modules\linear.py", line 101, in __init__
    self.reset_parameters()
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\torch\nn\modules\linear.py", line 107, in reset_parameters
    init.kaiming_uniform_(self.weight, a=math.sqrt(5))
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\torch\nn\init.py", line 412, in kaiming_uniform_
    return tensor.uniform_(-bound, bound)
  File "C:\Users\Tyler\Desktop\SD\Amd T\stable-diffusion-webui-directml\venv\lib\site-packages\torch\_decomp\decompositions.py", line 1958, in uniform_
    return self.copy_((high - low) * torch.rand_like(self) + low)
TypeError: unsupported operand type(s) for *: 'float' and 'Tensor'
Generating class images 0/31::   0%|                                                            | 0/31 [00:01<?, ?it/s]
Restored system models.
Duration: 00:00:03

