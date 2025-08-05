"""Simulation save/load management system."""

import json
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import pandas as pd

from .portfolio_engine import Trade, PortfolioSnapshot

logger = logging.getLogger(__name__)


class SimulationSave:
    """Container for simulation save data."""
    
    def __init__(self, name: str, config: Dict, results: Dict, 
                 timestamp: datetime = None):
        self.name = name
        self.config = config
        self.results = results
        self.timestamp = timestamp or datetime.now()
        self.description = ""
        self.tags = []
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'config': self.config,
            'results': self._serialize_results(),
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'tags': self.tags
        }
    
    def _serialize_results(self) -> Dict:
        """Serialize results for JSON storage."""
        serialized = {}
        
        for key, value in self.results.items():
            if key == 'trades_df' and isinstance(value, pd.DataFrame):
                serialized[key] = value.to_dict('records')
            elif key == 'portfolio_df' and isinstance(value, pd.DataFrame):
                serialized[key] = {
                    'data': value.to_dict('records'),
                    'index': value.index.strftime('%Y-%m-%d').tolist() if hasattr(value.index, 'strftime') else value.index.tolist()
                }
            elif isinstance(value, (pd.Timestamp, datetime)):
                serialized[key] = value.isoformat()
            elif isinstance(value, pd.DataFrame):
                serialized[key] = value.to_dict('records')
            else:
                serialized[key] = value
        
        return serialized
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationSave':
        """Create from dictionary loaded from JSON."""
        save = cls(
            name=data['name'],
            config=data['config'],
            results={},
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
        save.description = data.get('description', '')
        save.tags = data.get('tags', [])
        
        # Deserialize results
        save.results = save._deserialize_results(data['results'])
        
        return save
    
    def _deserialize_results(self, results_data: Dict) -> Dict:
        """Deserialize results from JSON data."""
        deserialized = {}
        
        for key, value in results_data.items():
            if key == 'trades_df' and isinstance(value, list):
                deserialized[key] = pd.DataFrame(value)
                # Convert date columns back to datetime
                if not deserialized[key].empty:
                    for col in ['entry_date', 'exit_date']:
                        if col in deserialized[key].columns:
                            deserialized[key][col] = pd.to_datetime(deserialized[key][col])
            elif key == 'portfolio_df' and isinstance(value, dict):
                if 'data' in value and 'index' in value:
                    df = pd.DataFrame(value['data'])
                    df.index = pd.to_datetime(value['index'])
                    deserialized[key] = df
                else:
                    deserialized[key] = pd.DataFrame(value)
            elif isinstance(value, str) and self._is_iso_datetime(value):
                deserialized[key] = datetime.fromisoformat(value)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                deserialized[key] = pd.DataFrame(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    def _is_iso_datetime(self, value: str) -> bool:
        """Check if string is ISO datetime format."""
        try:
            datetime.fromisoformat(value)
            return True
        except (ValueError, TypeError):
            return False


class SimulationManager:
    """Manages saving and loading of simulation states."""
    
    def __init__(self, save_dir: str = "simulation_saves"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.save_dir / "json").mkdir(exist_ok=True)
        (self.save_dir / "pickle").mkdir(exist_ok=True)
        
        logger.info(f"SimulationManager initialized with save directory: {self.save_dir}")
    
    def save_simulation(self, name: str, config: Dict, results: Dict, 
                       description: str = "", tags: List[str] = None,
                       format: str = "json") -> bool:
        """
        Save simulation results.
        
        Args:
            name: Name for the saved simulation
            config: Configuration used for the simulation
            results: Simulation results
            description: Optional description
            tags: Optional tags for categorization
            format: Save format ('json' or 'pickle')
        
        Returns:
            True if saved successfully
        """
        try:
            # Create save object
            save = SimulationSave(name, config, results)
            save.description = description
            save.tags = tags or []
            
            # Sanitize filename
            safe_name = self._sanitize_filename(name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_name}_{timestamp}"
            
            if format == "json":
                filepath = self.save_dir / "json" / f"{filename}.json"
                with open(filepath, 'w') as f:
                    json.dump(save.to_dict(), f, indent=2, default=str)
            
            elif format == "pickle":
                filepath = self.save_dir / "pickle" / f"{filename}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(save, f)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Simulation saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save simulation '{name}': {e}")
            return False
    
    def load_simulation(self, filename: str) -> Optional[SimulationSave]:
        """
        Load simulation from file.
        
        Args:
            filename: Name of the file to load (with or without extension)
        
        Returns:
            SimulationSave object or None if failed
        """
        try:
            # Try different paths and extensions
            possible_paths = [
                self.save_dir / "json" / f"{filename}.json",
                self.save_dir / "json" / filename,
                self.save_dir / "pickle" / f"{filename}.pkl",
                self.save_dir / "pickle" / filename,
                self.save_dir / filename
            ]
            
            filepath = None
            for path in possible_paths:
                if path.exists():
                    filepath = path
                    break
            
            if not filepath:
                logger.error(f"Simulation file not found: {filename}")
                return None
            
            # Load based on extension
            if filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return SimulationSave.from_dict(data)
            
            elif filepath.suffix == '.pkl':
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            
            else:
                logger.error(f"Unsupported file format: {filepath.suffix}")
                return None
            
        except Exception as e:
            logger.error(f"Failed to load simulation '{filename}': {e}")
            return None
    
    def list_simulations(self) -> List[Dict[str, Any]]:
        """
        List all saved simulations.
        
        Returns:
            List of simulation metadata
        """
        simulations = []
        
        # Scan JSON files
        json_dir = self.save_dir / "json"
        if json_dir.exists():
            for filepath in json_dir.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    simulations.append({
                        'filename': filepath.stem,
                        'name': data.get('name', filepath.stem),
                        'timestamp': data.get('timestamp', ''),
                        'description': data.get('description', ''),
                        'tags': data.get('tags', []),
                        'format': 'json',
                        'size_mb': filepath.stat().st_size / (1024 * 1024),
                        'filepath': str(filepath)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {filepath}: {e}")
        
        # Scan pickle files
        pickle_dir = self.save_dir / "pickle"
        if pickle_dir.exists():
            for filepath in pickle_dir.glob("*.pkl"):
                try:
                    # For pickle files, we need to load to get metadata
                    with open(filepath, 'rb') as f:
                        save = pickle.load(f)
                    
                    simulations.append({
                        'filename': filepath.stem,
                        'name': save.name,
                        'timestamp': save.timestamp.isoformat(),
                        'description': save.description,
                        'tags': save.tags,
                        'format': 'pickle',
                        'size_mb': filepath.stat().st_size / (1024 * 1024),
                        'filepath': str(filepath)
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {filepath}: {e}")
        
        # Sort by timestamp descending
        simulations.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return simulations
    
    def delete_simulation(self, filename: str) -> bool:
        """
        Delete a saved simulation.
        
        Args:
            filename: Name of the file to delete
        
        Returns:
            True if deleted successfully
        """
        try:
            # Find the file
            possible_paths = [
                self.save_dir / "json" / f"{filename}.json",
                self.save_dir / "json" / filename,
                self.save_dir / "pickle" / f"{filename}.pkl",
                self.save_dir / "pickle" / filename
            ]
            
            deleted = False
            for path in possible_paths:
                if path.exists():
                    path.unlink()
                    logger.info(f"Deleted simulation: {path}")
                    deleted = True
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete simulation '{filename}': {e}")
            return False
    
    def export_simulation(self, filename: str, export_path: str) -> bool:
        """
        Export simulation to a different location.
        
        Args:
            filename: Name of the simulation to export
            export_path: Path where to export the file
        
        Returns:
            True if exported successfully
        """
        try:
            save = self.load_simulation(filename)
            if not save:
                return False
            
            export_path = Path(export_path)
            
            # Export as JSON for portability
            with open(export_path, 'w') as f:
                json.dump(save.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Simulation exported to: {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export simulation '{filename}': {e}")
            return False
    
    def import_simulation(self, import_path: str) -> bool:
        """
        Import simulation from external file.
        
        Args:
            import_path: Path to the file to import
        
        Returns:
            True if imported successfully
        """
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                logger.error(f"Import file not found: {import_path}")
                return False
            
            # Load the data
            with open(import_path, 'r') as f:
                data = json.load(f)
            
            save = SimulationSave.from_dict(data)
            
            # Save in our format
            return self.save_simulation(
                save.name,
                save.config,
                save.results,
                save.description,
                save.tags
            )
            
        except Exception as e:
            logger.error(f"Failed to import simulation from '{import_path}': {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            'total_simulations': 0,
            'json_files': 0,
            'pickle_files': 0,
            'total_size_mb': 0.0,
            'oldest_simulation': None,
            'newest_simulation': None
        }
        
        simulations = self.list_simulations()
        stats['total_simulations'] = len(simulations)
        
        for sim in simulations:
            if sim['format'] == 'json':
                stats['json_files'] += 1
            else:
                stats['pickle_files'] += 1
            
            stats['total_size_mb'] += sim['size_mb']
        
        if simulations:
            stats['oldest_simulation'] = simulations[-1]['timestamp']
            stats['newest_simulation'] = simulations[0]['timestamp']
        
        return stats
    
    def cleanup_old_saves(self, keep_count: int = 50) -> int:
        """
        Clean up old simulation saves, keeping only the most recent ones.
        
        Args:
            keep_count: Number of simulations to keep
        
        Returns:
            Number of files deleted
        """
        simulations = self.list_simulations()
        
        if len(simulations) <= keep_count:
            return 0
        
        # Delete oldest simulations
        to_delete = simulations[keep_count:]
        deleted_count = 0
        
        for sim in to_delete:
            if self.delete_simulation(sim['filename']):
                deleted_count += 1
        
        logger.info(f"Cleanup completed: deleted {deleted_count} old simulation saves")
        return deleted_count
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        
        return filename.strip()