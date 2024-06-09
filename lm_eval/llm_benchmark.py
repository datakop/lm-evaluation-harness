import sys
import os
import json
import logging

from dataclasses import dataclass

import nltk
from transformers import PreTrainedModel

from lm_eval.tasks import get_task_dict
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMBenchmarkConfig:
    experiment_name: str
    batch_size: int
    device: str
    samples_limit_per_dataset: int
    task_list: list
    path_to_results: str
    fewshot_random_seed: int = 1234

class LLMBenchmark:
    def __init__(self, config: LLMBenchmarkConfig):
        self.config = config
        self.task_dict = self.load_tasks()

    def load_tasks(self):
        task_dict = get_task_dict(self.config.task_list)

        nltk.download('punkt')
        gen_kwargs = None

        for task_name in task_dict.keys():
            print(task_name)
            task_obj = task_dict[task_name]
            if isinstance(task_obj, tuple):
                _, task_obj = task_obj
                if task_obj is None:
                    continue

            if task_obj.get_config("output_type") == "generate_until":
                if gen_kwargs is not None:
                    task_obj.set_config(
                        key="generation_kwargs", value=gen_kwargs, update=True
                    )

            task_obj.set_config(key="num_fewshot", value=task_obj.get_config("num_fewshot"))
            task_obj.set_fewshot_seed(seed=self.config.fewshot_random_seed)
        
        logger.info("Tasks loaded successfully.")
        return task_dict

    def wrap_model(self, model: PreTrainedModel):
        model_args = {'pretrained': model, 'trust_remote_code': True}
        model_args2 = {'batch_size': self.config.batch_size, 'device': self.config.device}
        logger.info("Model wrapped successfully.")
        return HFLM(**model_args, **model_args2)

    def run_benchmark(self, model: PreTrainedModel):
        lm = self.wrap_model(model)
        results = {}
        for key in self.task_dict.keys():
            logger.info(f'Running benchmark for task: {key}')
            result = evaluator.evaluate(
                lm=lm,
                task_dict={k: v for k, v in self.task_dict.items() if k in [key]},
                limit=self.config.samples_limit_per_dataset,
                cache_requests=True,
            )
            results[key] = result["results"]
            logger.info(f'Results for task {key}: {result["results"]}')
        return results

    def save_results(self, results):
        path_to_results = os.path.join(self.config.path_to_results, self.config.experiment_name)
        os.makedirs(path_to_results, exist_ok=True)

        with open(f"{path_to_results}/results.json", 'w') as f:
            json.dump(results, f, indent=4)
        with open(f"{path_to_results}/full_output.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {path_to_results}")

    def run(self, model: PreTrainedModel):
        results = self.run_benchmark(model)
        self.save_results(results)
        return results