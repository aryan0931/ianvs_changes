Here is the `README.md` file you requested in Markdown format:

```markdown
# Ianvs MLLM Benchmarking

This document outlines the modifications and additions made to the Ianvs project to support the benchmarking of Multimodal Large Language Models (MLLM).

## Introduction

Ianvs aims to test the performance of distributed synergy AI solutions following recognized standards. This extension focuses on enabling Ianvs to benchmark Multimodal Large Language Models, which require handling and evaluating data from multiple modalities (e.g., text, images, audio).

## Modifications

### 1. Dataset Handling

The `Dataset` class has been updated to handle multiple types of data such as text, images, and audio.

#### Example:
```python
class Dataset:
    def __init__(self, config):
        self.train_urls = config.get('train_urls', [])
        self.test_urls = config.get('test_urls', [])
        self.modalities = config.get('modalities', [])  # New attribute to handle different data types

    def process_datasets(self):
        for modality in self.modalities:
            for train_url in self.train_urls:
                self.process_dataset(train_url, modality)
            for test_url in self.test_urls:
                self.process_dataset(test_url, modality)

    def process_dataset(self, url, modality):
        # Process dataset based on modality (text, image, audio, etc.)
        pass
```

### 2. Test Environment Configuration

The `TestEnv` class has been updated to support configurations for multimodal datasets and specific metrics for large language models.

#### Example:
```python
class TestEnv:
    def __init__(self, config):
        self.model_eval = {
            "model_metric": {
                "mode": "",
                "name": "",
                "url": "",
            },
            "threshold": 0.9,
            "operator": ">"
        }
        self.metrics = []
        self.incremental_rounds = 2
        self.datasets = []  # Changed from single dataset to a list of datasets
        self.modalities = []  # New attribute for handling multiple modalities
        self._parse_config(config)

    def _check_fields(self):
        if not self.metrics:
            raise ValueError(f"not found testenv metrics({self.metrics}).")
        if not isinstance(self.incremental_rounds, int) or self.incremental_rounds < 2:
            raise ValueError(f"testenv incremental_rounds(value={self.incremental_rounds})"
                             f" must be int type and not less than 2.")

    def _parse_config(self, config):
        config_dict = config[str.lower(TestEnv.__name__)]
        for k, v in config_dict.items():
            if k == str.lower(Dataset.__name__):
                self.datasets.append(Dataset(v))
            elif k == "modalities":
                self.modalities = v
            else:
                if k in self.__dict__:
                    self.__dict__[k] = v

        self._check_fields()

    def prepare(self):
        try:
            for dataset in self.datasets:
                dataset.process_datasets()
        except Exception as err:
            raise RuntimeError(f"prepare dataset failed, error: {err}.") from err
```

### 3. New Benchmarking Configuration

A new benchmarking job configuration file specific to MLLM has been created.

#### Example:
```yaml
benchmarkingjob:
  name: "mllm_benchmarking_job"
  workspace: "/ianvs/multimodal_language_model_bench/workspace"
  testenv: "./examples/mllm_benchmark/testenv.yaml"
  test_object:
    type: "algorithms"
    algorithms:
      - name: "mllm_evaluation"
        url: "./examples/mllm_benchmark/algorithms/mllm_algorithm.yaml"
  rank:
    sort_by: [ { "accuracy": "descend" }, { "f1_score": "descend" } ]
    visualization:
      mode: "selected_only"
      method: "print_table"
    selected_dataitem:
      paradigms: [ "all" ]
      modules: [ "all" ]
      hyperparameters: [ "all" ]
      metrics: [ "accuracy", "f1_score" ]
```

### 4. Algorithm Configuration

A new algorithm configuration file for the MLLM benchmark has been created.

#### Example:
```yaml
algorithm:
  paradigm_type: "multimodal_learning"
  initial_model_url: ""
  modules:
    - type: "basemodel"
      name: "MLLM_base"
      url: "./examples/mllm_benchmark/testalgorithms/mllm_base_model.py"
      hyperparameters:
        - config:
            values:
              - "./examples/mllm_benchmark/resource/MLLM_config.py"
        - work_dir:
            values:
              - "./examples/mllm_benchmark/work_dir"
        - resource_dir:
            values:
              - "./examples/mllm_benchmark/resource"
```

### 5. Metrics Update

The `metrics.py` file has been updated to include new metric functions for evaluating multimodal models.

#### Example:
```python
def multimodal_accuracy_func(system_metric_info: dict):
    info = system_metric_info.get("multimodal_accuracy")
    correct_predictions = info.get("correct_predictions", 0)
    total_predictions = info.get("total_predictions", 1)
    return round(correct_predictions / total_predictions, 4)

def cross_modal_retrieval_func(system_metric_info: dict):
    info = system_metric_info.get("cross_modal_retrieval")
    retrieval_score = info.get("retrieval_score", 0)
    return retrieval_score

def get_metric_func(metric_dict: dict):
    name = metric_dict.get("name")
    url = metric_dict.get("url")
    if url:
        try:
            load_module(url)
            metric_func = ClassFactory.get_cls(
                type_name=ClassType.GENERAL, t_cls_name=name)
            return name, metric_func
        except Exception as err:
            raise RuntimeError(
                f"get metric func(url={url}) failed, error: {err}.") from err

    metric_func_map = {
        'multimodal_accuracy': multimodal_accuracy_func,
        'cross_modal_retrieval': cross_modal_retrieval_func,
    }
    
    return name, metric_func_map.get(name, getattr(sys.modules[__name__], str.lower(name) + "_func"))
```

