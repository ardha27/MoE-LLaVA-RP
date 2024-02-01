INPUT_SCHEMA = {
    'model_path': {
        'type': str,
        'required': False,
        'default': 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e'
    },
    'image': {
        'type': str,
        'required': True
    },
    'prompt': {
        'type': str,
        'required': True
    },
    'conv_mode': {
        'type': str,
        'required': False,
        'default': 'phi'  # Default to 'phi', change as needed
    },
    'temperature': {
        'type': float,
        'required': False,
        'default': 0.2
    },
    'max_new_tokens': {
        'type': int,
        'required': False,
        'default': 1024
    },
    'load_8bit': {
        'type': bool,
        'required': False,
        'default': False
    },
    'load_4bit': {
        'type': bool,
        'required': False,
        'default': False
    }
}
