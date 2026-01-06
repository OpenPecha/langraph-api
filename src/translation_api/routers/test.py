from fastapi import APIRouter, Body
from fastapi import Body
from pydantic import BaseModel


# Import get_translation_with_context from the appropriate module
from ..utils.translation_compare import get_translation_with_context

class TranslationRequest(BaseModel):
    text: str

router = APIRouter(prefix="/test", tags=["Test"])
@router.post("/compare-translation")
async def compare_translation(request: TranslationRequest = Body(...)):
    """
    Compare translation with and without context and return the result.
    """
    result = get_translation_with_context(request.text)
    print(result)
    return result
