# modules/model_cache.py

import logging
from pathlib import Path
import torch
import spacy
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

logger = logging.getLogger(__name__)

class ModelCache:
    """
    Singleton class to cache and reuse transformers and spaCy models.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelCache, cls).__new__(cls)
            cls._instance.models = {}
            cls._instance.tokenizers = {}
            cls._instance.nlp_models = {}
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.cache_dir = Path("data/model_cache")
            cls._instance.cache_dir.mkdir(parents=True, exist_ok=True)
            cls._instance.max_memory_usage = 0.8  # Maximum memory usage (80% of available)
            cls._instance.model_sizes = {}  # Track approximate model sizes
        return cls._instance

    # ✅ Do NOT override cache_dir here

    def _check_memory(self):
        """Check if we have enough memory to load a new model"""
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            return current_memory < self.max_memory_usage
        return True  # If no GPU, assume we have enough memory
        
    def _estimate_model_size(self, model):
        """Estimate model size in memory"""
        try:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return param_size + buffer_size
        except Exception:
            return 0
            
    def _clear_oldest_model(self):
        """Clear the oldest model if memory is low"""
        if not self.models:
            return
            
        oldest_model = min(self.models.items(), key=lambda x: x[1]['timestamp'])
        model_name = oldest_model[0]
        print(f"Clearing oldest model from cache: {model_name}")
        del self.models[model_name]
        if model_name in self.tokenizers:
            del self.tokenizers[model_name]
        torch.cuda.empty_cache()
        
    def get_transformer_model(self, model_name):
        """Get or load a transformer model with improved error handling"""
        try:
            if model_name in self.models:
                self.models[model_name]['timestamp'] = time.time()
                return self.models[model_name]['model']
                
            if not self._check_memory():
                self._clear_oldest_model()
                
            print(f"Loading transformer model: {model_name}")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                resume_download=True
            ).to(self.device)
            model.eval()
            
            # Track model size and timestamp
            size = self._estimate_model_size(model)
            self.models[model_name] = {
                'model': model,
                'size': size,
                'timestamp': time.time()
            }
            
            return model
            
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            # Try to load from cache if available
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                ).to(self.device)
                model.eval()
                return model
            except Exception as cache_error:
                print(f"Failed to load from cache: {str(cache_error)}")
                raise RuntimeError(f"Could not load model {model_name}: {str(e)}")

    def get_tokenizer(self, model_name):
        """Get or load a tokenizer with improved error handling"""
        try:
            if model_name in self.tokenizers:
                return self.tokenizers[model_name]
                
            print(f"Loading tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=False,
                resume_download=True
            )
            self.tokenizers[model_name] = tokenizer
            return tokenizer
            
        except Exception as e:
            print(f"Error loading tokenizer {model_name}: {str(e)}")
            # Try to load from cache
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir,
                    local_files_only=True
                )
                return tokenizer
            except Exception as cache_error:
                print(f"Failed to load tokenizer from cache: {str(cache_error)}")
                raise RuntimeError(f"Could not load tokenizer {model_name}: {str(e)}")

    def get_spacy_model(self, model_name):
        """Get or load a spaCy model with improved error handling"""
        try:
            if model_name in self.nlp_models:
                return self.nlp_models[model_name]
                
            print(f"Loading spaCy model: {model_name}")
            try:
                nlp = spacy.load(model_name)
            except OSError:
                print(f"Downloading spaCy model: {model_name}")
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                
            nlp.max_length = 3_500_000
            self.nlp_models[model_name] = nlp
            return nlp
            
        except Exception as e:
            print(f"Error loading spaCy model {model_name}: {str(e)}")
            # Try to load a smaller model as fallback
            if model_name == "en_core_web_sm":
                try:
                    print("Attempting to load en_core_web_sm as fallback")
                    nlp = spacy.blank("en")
                    nlp.add_pipe("sentencizer")
                    return nlp
                except Exception as fallback_error:
                    print(f"Fallback model loading failed: {str(fallback_error)}")
            raise RuntimeError(f"Could not load spaCy model {model_name}: {str(e)}")

    def clear_cache(self):
        """Clear all cached models and free memory"""
        print("Clearing model cache")
        self.models.clear()
        self.tokenizers.clear()
        self.nlp_models.clear()
        self.model_sizes.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU memory cleared")

# Instantiate globally
model_cache = ModelCache()
