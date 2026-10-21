class FrameworkError(Exception):
    """Base framework exception."""


class DatasetValidationError(FrameworkError):
    """Raised when dataset records are invalid."""


class UnsupportedConfigurationError(FrameworkError):
    """Raised when model/config combination is unsupported."""


class PromptBuildError(FrameworkError):
    """Raised when a prompt cannot be constructed."""


class PromptTemplateNotFoundError(FrameworkError):
    """Raised when a (template_id, template_version) does not resolve."""


class ModelRegistrationError(FrameworkError):
    """Raised when registry/factory issues occur."""


class ConfigError(FrameworkError):
    """Raised when YAML or runtime config is invalid."""


class CacheClientError(FrameworkError):
    """Raised when cache-aware client construction fails."""
