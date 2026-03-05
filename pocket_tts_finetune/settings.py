from dataclasses import dataclass


@dataclass(frozen=True)
class Paths:
    vol: str = "/vol"
    dataset_path: str = "/vol/dataset_pocket"
    lora_path: str = "/vol/pocket_lora"
    merged_path: str = "/vol/pocket_merged"
    voices_path: str = "/vol/voices"
