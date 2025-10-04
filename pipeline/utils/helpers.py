""""""

Utility functions for the pipeline.Utility functions for the pipeline.

Only contains functions that are actually used in the codebase.Only contains functions that are actually used in the codebase.

""""""



import loggingimport logging

import timeimport time

from typing import Callablefrom typing import Callable

from functools import wrapsfrom functools import wraps



def setup_logging(level: str = "INFO", format_string: str = None) -> None:def setup_logging(level: str = "INFO", format_string: str = None) -> None:

    """    """

    Set up logging for the pipeline.    Set up logging for the pipeline.

        

    Args:    Args:

        level: Logging level (DEBUG, INFO, WARNING, ERROR)        level: Logging level (DEBUG, INFO, WARNING, ERROR)

        format_string: Custom format string for log messages        format_string: Custom format string for log messages

    """    """

    if format_string is None:    if format_string is None:

        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        

    logging.basicConfig(    logging.basicConfig(

        level=getattr(logging, level.upper()),        level=getattr(logging, level.upper()),

        format=format_string,        format=format_string,

        handlers=[logging.StreamHandler()]        handlers=[logging.StreamHandler()]

    )    )



def time_function(func: Callable) -> Callable:def time_function(func: Callable) -> Callable:

    """    """

    Decorator to time function execution.    Decorator to time function execution.

        

    Args:    Args:

        func: Function to time        func: Function to time

                

    Returns:    Returns:

        Wrapped function that logs execution time        Wrapped function that logs execution time

    """    """

    @wraps(func)    @wraps(func)

    def wrapper(*args, **kwargs):    def wrapper(*args, **kwargs):

        start_time = time.time()        start_time = time.time()

        result = func(*args, **kwargs)        result = func(*args, **kwargs)

        end_time = time.time()        end_time = time.time()

                

        logger = logging.getLogger(func.__module__)        logger = logging.getLogger(func.__module__)

        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")

                

        return result        return result

    return wrapper    return wrapper



def create_progress_callback(total: int, description: str = "Processing") -> Callable:def batch_process(items: List[Any], batch_size: int, process_func: Callable) -> List[Any]:

    """    """

    Create a progress callback function.    Process items in batches.

        

    Args:    Args:

        total: Total number of items to process        items: List of items to process

        description: Description of the process        batch_size: Size of each batch

                process_func: Function to process each batch

    Returns:        

        Callback function that can be called with current progress    Returns:

    """        List of processed results

    def callback(current: int) -> None:    """

        percentage = (current / total) * 100 if total > 0 else 0    results = []

        print(f"\r{description}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)    

            for i in range(0, len(items), batch_size):

        if current >= total:        batch = items[i:i + batch_size]

            print()  # New line when complete        batch_results = process_func(batch)

            

    return callback        if isinstance(batch_results, list):
            results.extend(batch_results)
        else:
            results.append(batch_results)
    
    return results

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    return numerator / denominator if denominator != 0 else default

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length with optional suffix.
    
    Args:
        text: Text to truncate
        max_length: Maximum length including suffix
        suffix: Suffix to add if text is truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def calculate_overlap(text1: str, text2: str) -> float:
    """
    Calculate the overlap between two texts as a ratio.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap ratio between 0 and 1
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return safe_divide(intersection, union)

def validate_config(config: Dict[str, Any], required_keys: List[str], 
                   optional_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: Keys that must be present
        optional_keys: Keys that are optional
        
    Returns:
        Validated configuration
        
    Raises:
        ValueError: If required keys are missing or invalid keys are present
    """
    optional_keys = optional_keys or []
    
    # Check for missing required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")
    
    # Check for unexpected keys
    allowed_keys = set(required_keys + optional_keys)
    unexpected_keys = [key for key in config.keys() if key not in allowed_keys]
    if unexpected_keys:
        raise ValueError(f"Unexpected configuration keys: {unexpected_keys}")
    
    return config

def create_progress_callback(total: int, description: str = "Processing") -> Callable:
    """
    Create a progress callback function.
    
    Args:
        total: Total number of items to process
        description: Description of the process
        
    Returns:
        Callback function that can be called with current progress
    """
    def callback(current: int) -> None:
        percentage = (current / total) * 100 if total > 0 else 0
        print(f"\r{description}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
        
        if current >= total:
            print()  # New line when complete
    
    return callback

def estimate_memory_usage(num_vectors: int, vector_dim: int, 
                         bytes_per_float: int = 4) -> Dict[str, float]:
    """
    Estimate memory usage for vector storage.
    
    Args:
        num_vectors: Number of vectors
        vector_dim: Dimension of each vector
        bytes_per_float: Bytes per floating point number
        
    Returns:
        Dictionary with memory estimates in different units
    """
    total_bytes = num_vectors * vector_dim * bytes_per_float
    
    return {
        "bytes": total_bytes,
        "kb": total_bytes / 1024,
        "mb": total_bytes / (1024 * 1024),
        "gb": total_bytes / (1024 * 1024 * 1024)
    }

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

def create_config_from_dict(base_config: Dict[str, Any], 
                           overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration by merging base config with overrides.
    
    Args:
        base_config: Base configuration dictionary
        overrides: Override values
        
    Returns:
        Merged configuration
    """
    config = base_config.copy()
    config.update(overrides)
    return config