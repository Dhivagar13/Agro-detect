"""Settings Manager for AgroDetect AI"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class GeneralSettings:
    """General application settings"""
    auto_save: bool = True
    notifications: bool = True
    sound_alerts: bool = False
    max_history: int = 1000
    auto_cleanup: bool = False
    cleanup_days: int = 30
    cache_enabled: bool = True
    gpu_acceleration: bool = True
    language: str = "English"
    timezone: str = "UTC"
    date_format: str = "MM/DD/YYYY"


@dataclass
class ModelSettings:
    """Model configuration settings"""
    confidence_threshold: float = 50.0
    batch_inference: bool = False
    batch_size: int = 8
    model_path: str = "models/plant_disease_model.h5"
    class_names_path: str = "models/class_names.json"
    groq_api_key: str = ""


@dataclass
class AppearanceSettings:
    """Appearance settings"""
    theme: str = "Default (Purple)"
    sidebar_default: str = "Expanded"
    chart_style: str = "Modern"
    show_animations: bool = True
    compact_mode: bool = False


class SettingsManager:
    """
    Manages application settings with persistence
    """
    
    def __init__(self, settings_file: str = "config/user_settings.json"):
        """
        Initialize Settings Manager
        
        Args:
            settings_file: Path to settings file
        """
        self.settings_file = Path(settings_file)
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize settings
        self.general = GeneralSettings()
        self.model = ModelSettings()
        self.appearance = AppearanceSettings()
        
        # Load existing settings
        self.load()
    
    def load(self) -> bool:
        """
        Load settings from file
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.settings_file.exists():
            return False
        
        try:
            with open(self.settings_file, 'r') as f:
                data = json.load(f)
            
            # Load general settings
            if 'general' in data:
                for key, value in data['general'].items():
                    if hasattr(self.general, key):
                        setattr(self.general, key, value)
            
            # Load model settings
            if 'model' in data:
                for key, value in data['model'].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)
            
            # Load appearance settings
            if 'appearance' in data:
                for key, value in data['appearance'].items():
                    if hasattr(self.appearance, key):
                        setattr(self.appearance, key, value)
            
            return True
        
        except Exception as e:
            print(f"Error loading settings: {e}")
            return False
    
    def save(self) -> bool:
        """
        Save settings to file
        
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            data = {
                'general': asdict(self.general),
                'model': asdict(self.model),
                'appearance': asdict(self.appearance)
            }
            
            with open(self.settings_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        
        except Exception as e:
            print(f"Error saving settings: {e}")
            return False
    
    def update_general(self, **kwargs) -> bool:
        """
        Update general settings
        
        Args:
            **kwargs: Settings to update
        
        Returns:
            True if updated successfully
        """
        for key, value in kwargs.items():
            if hasattr(self.general, key):
                setattr(self.general, key, value)
        
        return self.save()
    
    def update_model(self, **kwargs) -> bool:
        """
        Update model settings
        
        Args:
            **kwargs: Settings to update
        
        Returns:
            True if updated successfully
        """
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        
        return self.save()
    
    def update_appearance(self, **kwargs) -> bool:
        """
        Update appearance settings
        
        Args:
            **kwargs: Settings to update
        
        Returns:
            True if updated successfully
        """
        for key, value in kwargs.items():
            if hasattr(self.appearance, key):
                setattr(self.appearance, key, value)
        
        return self.save()
    
    def reset_to_defaults(self) -> bool:
        """
        Reset all settings to defaults
        
        Returns:
            True if reset successfully
        """
        self.general = GeneralSettings()
        self.model = ModelSettings()
        self.appearance = AppearanceSettings()
        
        return self.save()
    
    def get_all_settings(self) -> Dict[str, Any]:
        """
        Get all settings as dictionary
        
        Returns:
            Dictionary with all settings
        """
        return {
            'general': asdict(self.general),
            'model': asdict(self.model),
            'appearance': asdict(self.appearance)
        }
    
    def apply_confidence_threshold(self, confidence: float) -> bool:
        """
        Check if confidence meets threshold
        
        Args:
            confidence: Confidence score
        
        Returns:
            True if meets threshold
        """
        return confidence >= self.model.confidence_threshold
    
    def should_auto_save(self) -> bool:
        """Check if auto-save is enabled"""
        return self.general.auto_save
    
    def should_show_notifications(self) -> bool:
        """Check if notifications are enabled"""
        return self.general.notifications
    
    def get_max_history(self) -> int:
        """Get maximum history records"""
        return self.general.max_history
    
    def is_cache_enabled(self) -> bool:
        """Check if caching is enabled"""
        return self.general.cache_enabled
    
    def get_date_format(self) -> str:
        """Get date format"""
        return self.general.date_format


# Global settings instance
_settings_manager = None


def get_settings_manager() -> SettingsManager:
    """
    Get global settings manager instance
    
    Returns:
        SettingsManager instance
    """
    global _settings_manager
    
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    
    return _settings_manager
