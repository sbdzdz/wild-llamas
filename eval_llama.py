from opencompass.models import TurboMindModelwithChatTemplate
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets


api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

datasets = [
    *mmlu_datasets,
    *mmlu_pro_datasets,
    *math_datasets,
    *gpqa_datasets,
]

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr="eval_model",
        path="models/eval_model",
        engine_config=dict(max_batch_size=16, tp=1),
        gen_config=dict(top_k=1, temperature=1e-6, top_p=0.9, max_new_tokens=4096),
        max_seq_len=8192,
        max_out_len=4096,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
        stop_words=["<|end_of_text|>", "<|eot_id|>"],
    )
]
