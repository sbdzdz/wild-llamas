from opencompass.models import TurboMindModelwithChatTemplate
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen import mmlu_datasets
    from opencompass.configs.datasets.mmlu_pro.mmlu_pro_gen import mmlu_pro_datasets
    from opencompass.configs.datasets.math.math_500_gen import math_datasets
    from opencompass.configs.datasets.gpqa.gpqa_gen import gpqa_datasets

BATCH_SIZE = None  # Replaced at runtime for evaluation batch size
DATASET_FRACTION = None  # Replaced at runtime for partial evaluations
datasets = []  # Populated at runtime

if DATASET_FRACTION is not None:
    for d in datasets:
        d.setdefault("reader_cfg", {})
        d["reader_cfg"].update({"test_range": DATASET_FRACTION})

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr="eval_model",
        path="models/eval_model",
        engine_config=dict(max_batch_size=BATCH_SIZE, tp=1),
        gen_config=dict(temperature=0.9, top_k=20, top_p=0.8, max_new_tokens=2048),
        max_seq_len=8192,
        max_out_len=2048,
        batch_size=BATCH_SIZE,
        run_cfg=dict(num_gpus=1),
        stop_words=["<|end_of_text|>", "<|eot_id|>"],
    )
]
