from opencompass.models import TurboMindModelwithChatTemplate
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
    from opencompass.configs.datasets.aime2024.aime2024_gen_17d799 import aime2024_datasets

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

datasets = [
    *math_datasets,
    # *aime2024_datasets,
]

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr="merged_model",
        path="models/merged_model",
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(do_sample=True, top_k=0, temperature=0.6, top_p=0.9, max_new_tokens=32768),
        max_seq_len=49152,  # 16384 + 32768
        max_out_len=32768,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        stop_words=["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"],
    )
]

work_dir = "outputs-reasoning/merged-llama3-instruct"
