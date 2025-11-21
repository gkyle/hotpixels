import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class UserPreferences:
    """User preferences and application state"""
    last_profile_path: Optional[str] = None
    last_darkframes_directory: Optional[str] = None
    last_image_directory: Optional[str] = None
    
    @classmethod
    def get_preferences_file_path(cls) -> Path:
        """Get the path to the preferences file"""
        # Store preferences in user's home directory
        home_dir = Path.home()
        preferences_dir = home_dir / ".hotpixels"
        preferences_dir.mkdir(exist_ok=True)
        return preferences_dir / "preferences.json"
    
    @classmethod
    def load(cls) -> 'UserPreferences':
        """Load preferences from file, creating defaults if file doesn't exist"""
        preferences_file = cls.get_preferences_file_path()
        
        if preferences_file.exists():
            try:
                with open(preferences_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Could not load preferences from {preferences_file}: {e}")
                print("Using default preferences.")
        
        # Return default preferences if file doesn't exist or can't be loaded
        return cls()
    
    def save(self) -> bool:
        """Save preferences to file"""
        preferences_file = self.get_preferences_file_path()
        
        try:
            # Ensure the directory exists
            preferences_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(preferences_file, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Warning: Could not save preferences to {preferences_file}: {e}")
            return False
    
    def update_last_profile_path(self, profile_path: Optional[str]) -> bool:
        """Update the last profile path and save preferences"""
        # Convert to absolute path if provided
        if profile_path:
            profile_path = str(Path(profile_path).resolve())
        
        self.last_profile_path = profile_path
        return self.save()
    
    def update_last_directory(self, path: str, directory_type: str) -> bool:
        """Update last used directory for different file types"""
        directory_path = str(Path(path).parent.resolve())
        
        if directory_type == "profile":
            self.last_profile_path = str(Path(path).resolve())
        elif directory_type == "darkframes":
            self.last_darkframes_directory = directory_path
        elif directory_type == "image":
            self.last_image_directory = directory_path
        
        return self.save()
    
    def get_last_directory(self, directory_type: str) -> Optional[str]:
        """Get the last used directory for a specific file type"""
        if directory_type == "darkframes":
            return self.last_darkframes_directory
        elif directory_type == "image":
            return self.last_image_directory
        elif directory_type == "profile":
            if self.last_profile_path:
                return str(Path(self.last_profile_path).parent)
        
        return None
    
    def get_valid_last_profile_path(self) -> Optional[str]:
        """Get the last profile path if the file still exists"""
        if self.last_profile_path and Path(self.last_profile_path).exists():
            return self.last_profile_path
        return None


# Global preferences instance
_preferences: Optional[UserPreferences] = None


def get_preferences() -> UserPreferences:
    """Get the global preferences instance"""
    global _preferences
    if _preferences is None:
        _preferences = UserPreferences.load()
    return _preferences


def save_preferences() -> bool:
    """Save the current preferences to file"""
    if _preferences is not None:
        return _preferences.save()
    return False
