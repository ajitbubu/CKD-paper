"""
Configuration utility for loading environment variables and API keys.
"""
import os from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration class for managing environment variables."""

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')
    OPENAI_ORG_ID = os.getenv('OPENAI_ORG_ID')

    @classmethod
    def validate(cls):
        """Validate that required environment variables are set."""
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == 'your-openai-api-key-here':
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )

    @classmethod
    def get_openai_client(cls):
        """
        Get an initialized OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not configured
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. "
                "Install it with: pip install openai"
            )

        cls.validate()

        client_kwargs = {'api_key': cls.OPENAI_API_KEY}
        if cls.OPENAI_ORG_ID:
            client_kwargs['organization'] = cls.OPENAI_ORG_ID

        return OpenAI(**client_kwargs)


# Convenience function for getting OpenAI client
def get_openai_client():
    """
    Convenience function to get an initialized OpenAI client.

    Returns:
        OpenAI client instance

    Example:
        >>> from src.utils.config import get_openai_client
        >>> client = get_openai_client()
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """
    return Config.get_openai_client()
