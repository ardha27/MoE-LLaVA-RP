#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download


if __name__ == '__main__':
    model = os.getenv('MODEL', 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e')

    print(f'Downloading LLaVA model: {model}')
    snapshot_download(model)

