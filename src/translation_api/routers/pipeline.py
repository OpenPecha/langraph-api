"""Pipeline endpoints for multi-stage workflows."""

from fastapi import APIRouter, HTTPException, Depends
from collections import defaultdict
from ..models.pipeline import PipelineRequest, PipelineResponse
from ..workflows.translation_state import TranslationRequest
from ..schemas.glossary import GlossaryExtractionRequest
from ..models.standardization import AnalysisRequest, AnalysisResponse, StandardizationRequest, StandardizationResponse, RetranslationResponse, StandardizationInputItem
from ..models.glossary import Glossary
from ..models.model_router import get_model_router
from ..prompts.tibetan_buddhist import RETRANSLATION_PROMPT, GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT
from ..api.dependencies import router_limiter
import sys
import os

# Import from root level graph module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from graph import run_translation_workflow

router = APIRouter(prefix="/pipeline", tags=["Pipeline"])


@router.post("/run", response_model=PipelineResponse, dependencies=[Depends(router_limiter)])
async def run_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Run a customizable workflow by selecting stages and providing inputs.

    Stages (in order):
    - translate: uses existing translation workflow
    - extract_glossary: runs glossary extraction on results/items
    - analyze: runs standardization analysis
    - apply_standardization: applies standardization pairs with minimal-change retranslation
    """
    aggregate: PipelineResponse = PipelineResponse(metadata={"stages": request.stages})

    # Stage: translate
    if "translate" in request.stages:
        if not request.texts or not request.target_language:
            raise HTTPException(status_code=400, detail="texts and target_language are required for translate stage")
        workflow_request = TranslationRequest(
            texts=request.texts,
            target_language=request.target_language,
            model_name=request.model_name,
            text_type=request.text_type,
            batch_size=request.batch_size,
            model_params=request.model_params,
            user_rules=request.user_rules,
        )
        final_state = await run_translation_workflow(workflow_request)
        aggregate.results = final_state["final_results"]
        aggregate.metadata.update({"translation": final_state.get("metadata", {})})

    # Build items for downstream stages
    items_source = request.items
    if aggregate.results:
        # Convert TranslationResult list into StandardizationInputItem objects
        items_source = [
            StandardizationInputItem(
                original_text=r.original_text,  # type: ignore[attr-defined]
                translated_text=r.translated_text,  # type: ignore[attr-defined]
                glossary=getattr(r, "glossary", []),
            )
            for r in aggregate.results
        ]

    # Stage: extract_glossary
    if "extract_glossary" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for glossary extraction")
        glossary_req = GlossaryExtractionRequest(items=items_source, model_name=request.model_name, batch_size=request.batch_size)  # type: ignore[arg-type]
        # Reuse glossary extraction logic
        model_router = get_model_router()
        try:
            model = model_router.get_model(glossary_req.model_name)
            structured_model = model.with_structured_output(Glossary)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        prompts = [
            GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                text_pairs=f"Source: {item.original_text}\\nTranslated: {item.translated_text}\\n\\n"
            ) for item in items_source
        ]

        all_terms = []
        try:
            for i in range(0, len(prompts), glossary_req.batch_size):
                batch_prompts = prompts[i:i + glossary_req.batch_size]
                glossary_results = await structured_model.abatch(batch_prompts)
                for gloss in glossary_results:
                    if gloss and gloss.terms:
                        all_terms.extend(gloss.terms)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Glossary extraction failed: {str(e)}")
        
        aggregate.glossary = Glossary(terms=all_terms)

    # Stage: analyze
    if "analyze" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for analysis")
        # If we have a new glossary, enrich items with glossary terms
        if aggregate.glossary:
            enriched = []
            for item in items_source:
                terms = [t for t in aggregate.glossary.terms if (t.source_term in item.original_text or t.translated_term in item.translated_text) ]
                enriched.append(StandardizationInputItem(
                    original_text=item.original_text,
                    translated_text=item.translated_text,
                    glossary=terms
                ))
            items_source = enriched
        # Reuse analysis logic
        term_map = defaultdict(set)
        for item in items_source:
            if item.glossary:
                for term in item.glossary:
                    term_map[term.source_term].add(term.translated_term)
        
        inconsistent_terms = {
            term: list(translations)
            for term, translations in term_map.items()
            if len(translations) > 1
        }
        aggregate.inconsistent_terms = inconsistent_terms

    # Stage: apply_standardization
    if "apply_standardization" in request.stages:
        if not items_source:
            raise HTTPException(status_code=400, detail="No items available for standardization")
        if not request.standardization_pairs:
            raise HTTPException(status_code=400, detail="standardization_pairs is required for apply_standardization stage")
        # Reuse standardization logic
        model_router = get_model_router()
        try:
            model = model_router.get_model(request.model_name)
            structured_model = model.with_structured_output(RetranslationResponse)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        source_words_to_standardize = {pair.source_word for pair in request.standardization_pairs}
        rules_block = ""
        for pair in request.standardization_pairs:
            rules_block += f'- For the source term "{pair.source_word}", use the exact translation "{pair.standardized_translation}".\n'

        updated_items = []
        for item in items_source:
            # Check if this item's original text contains any of the words to be standardized
            if any(word in item.original_text for word in source_words_to_standardize):
                prompt = RETRANSLATION_PROMPT.format(
                    user_rules=request.user_rules or "No specific user rules provided.",
                    standardization_rules_block=rules_block,
                    original_text=item.original_text,
                    original_translation=item.translated_text
                )
                
                try:
                    response = await structured_model.ainvoke(prompt)
                    new_translation = response.new_translation

                    glossary_model = model.with_structured_output(Glossary)
                    glossary_prompt = GLOSSARY_EXTRACTION_POST_TRANSLATION_PROMPT.format(
                        text_pairs=f"Source: {item.original_text}\\nTranslated: {new_translation}\\n\\n"
                    )
                    glossary_result = await glossary_model.ainvoke(glossary_prompt)
                    if glossary_result and glossary_result.terms:
                        updated_glossary = glossary_result.terms
                    else:
                        updated_glossary = []
                    
                    updated_items.append(StandardizationInputItem(
                        original_text=item.original_text,
                        translated_text=new_translation,
                        glossary=updated_glossary
                    ))
                except Exception as e:
                    print(f"Failed to re-translate item: {e}")
                    updated_items.append(item)
            else:
                updated_items.append(item)
        
        aggregate.updated_items = updated_items

    return aggregate

