"""FastAPI application for Tibetan Buddhist text translation.

This module maintains backward compatibility by re-exporting the modular API structure.
The actual implementation has been refactored into a modular structure:
- Routers: src/translation_api/routers/
- Schemas: src/translation_api/schemas/
- Services: src/translation_api/services/
- Utils: src/translation_api/utils/
"""

# Re-export the app and create_app function from the new modular structure
from .api.main import app, create_app

# Re-export schemas for backward compatibility
from .schemas.translation import TranslationAPIRequest, TranslationAPIResponse, SingleTranslationRequest
from .schemas.system import HealthResponse
from .schemas.glossary import GlossaryExtractionRequest
from .schemas.dharmamitra import DharmamitraKnnRequest, DharmamitraGeminiRequest
from .schemas.ucca import UCCAErrorResponse

# Re-export dependencies
from .api.dependencies import router_limiter

# For backward compatibility with uvicorn.run("src.translation_api.api:app")
__all__ = ["app", "create_app", "router_limiter"]
