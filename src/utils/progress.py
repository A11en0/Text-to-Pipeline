"""
Progress bar utility module
"""
from tqdm import tqdm

class ProgressBar:
    """Progress bar utility class, provides a wrapper for tqdm"""
    
    def __init__(self, total, desc="Progress", unit="items", initial=0, disable=False):
        """
        Initialize the progress bar
        
        Args:
            total: Total number of items
            desc: Description text
            unit: Unit name
            initial: Initial completed count
            disable: Whether to disable the progress bar
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            initial=initial,
            disable=disable
        )
        
    def update(self, n=1):
        """Update progress"""
        self.pbar.update(n)
        
    def set_description(self, desc):
        """Set description text"""
        self.pbar.set_description(desc)
        
    def set_postfix(self, **kwargs):
        """Set postfix information"""
        self.pbar.set_postfix(**kwargs)
        
    def close(self):
        """Close the progress bar"""
        self.pbar.close()
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()