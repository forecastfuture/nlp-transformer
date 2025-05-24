import os
from dotenv import load_dotenv, find_dotenv
from dynaconf import Dynaconf

# 加载环境变量
# load_dotenv(find_dotenv())

settings = Dynaconf(
    envvar_prefix="WID",
    settings_files=['./basic_config/settings.yaml'],
)

# 定义模型
MODELS = (settings.deepseek_models.model_names + settings.openai_models.model_names +
          settings.ali_models.model_names + settings.doubao_models.model_names)
DEFAULT_MODEL = MODELS[settings.default_model_index]

