from opencompass.models import HuggingFacewithChatTemplate
from mmengine.config import read_base

with read_base():
    from opencompass.configs.datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets

api_meta_template = dict(
    round=[
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True),
    ],
    reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
)

datasets = [
    *mmlu_datasets,
    *cmmlu_datasets,
]

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="merged_model",
        path="models/merged_model",
        max_out_len=32,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        stop_words=["<|end_of_text|>", "<|eot_id|>"],
        meta_template=api_meta_template,
    )
]

work_dir = "outputs/merged-llama3-instruct"
