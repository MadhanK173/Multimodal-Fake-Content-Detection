from typing import Any, Dict

from .text_plugin import TextPlugin
from .image_plugin import ImagePlugin
from .audio_plugin import AudioPlugin
from .evidence_plugin import EvidencePlugin
from .result_plugin import ResultPlugin


class PluginManager:
    """
    Central manager for the Multimodal Fake Content Detection plugins.
    """

    def __init__(self):
        self.plugins = {}

        # Register the currently available plugin
        self.register_plugin(
            "text",
            TextPlugin()
        )
        self.register_plugin(
            "image",
            ImagePlugin()
        )
        self.register_plugin(
            "audio",
            AudioPlugin()
        )
        self.register_plugin(
            "evidence",
            EvidencePlugin()
        )
        self.register_plugin(
            "result",
            ResultPlugin()
        )

    def register_plugin(
        self,
        name: str,
        plugin: Any
    ) -> None:
        """
        Register a plugin with the manager.
        """
        self.plugins[name] = plugin

    def get_plugin(
        self,
        name: str
    ) -> Any:
        """
        Get a registered plugin.
        """
        plugin = self.plugins.get(name)

        if plugin is None:
            raise ValueError(
                f"Plugin '{name}' is not registered."
            )

        return plugin

    def list_plugins(self) -> Dict[str, Any]:
        """
        Return all registered plugins.
        """
        return self.plugins

    def has_plugin(
        self,
        name: str
    ) -> bool:
        """
        Check whether a plugin is registered.
        """
        return name in self.plugins

    def detect_plugin(
        self,
        modality: str
    ) -> Any:
        """
        Select the appropriate plugin based
        on the input modality.
        """

        modality = (
            modality or ""
        ).strip().lower()

        if modality not in self.plugins:
            raise ValueError(
                f"No plugin registered for modality: {modality}"
            )

        return self.get_plugin(modality)