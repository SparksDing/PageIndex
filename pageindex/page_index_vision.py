"""
page_index_vision.py

Vision-aware wrapper around page_index.
Replaces text-based LLM calls with multimodal (image) calls by:
  1. Rendering each PDF page to a base64 JPEG via PyMuPDF.
  2. Monkey-patching llm_completion / llm_acompletion in the utils module
     so every call transparently injects the relevant page images.
  3. Delegating to the existing page_index() pipeline unchanged.

Usage:
    from pageindex.page_index_vision import page_index_vision

    result = page_index_vision(
        doc="path/to/file.pdf",
        vision_model="openai/gpt-4o",        # multimodal model
        vision_api_base="https://...",
        vision_api_key="sk-...",
        toc_model="openai/deepseek-v3",      # text model for TOC steps (optional)
        # all other page_index() kwargs are forwarded as-is
        if_add_node_summary="yes",
        if_add_node_id="yes",
    )
"""

import asyncio
import base64
import os
import threading
from contextvars import ContextVar
from io import BytesIO
from typing import Optional

import pymupdf  # PyMuPDF

# ---------------------------------------------------------------------------
# Page image cache: physical_index (1-based) → base64 JPEG string
# Stored per-thread / per-run via a plain dict protected by a lock.
# ---------------------------------------------------------------------------
_page_images: dict[int, str] = {}
_page_images_lock = threading.Lock()

# ContextVar that holds the list of page indices the current LLM call should see.
# Set by the patched llm helpers before each call.
_current_page_indices: ContextVar[list[int]] = ContextVar("_current_page_indices", default=[])


def _render_pdf_to_images(pdf_path, zoom: float = 2.0) -> dict[int, str]:
    """Render every page of a PDF to a base64 JPEG string (1-based index)."""
    if isinstance(pdf_path, BytesIO):
        doc = pymupdf.open(stream=pdf_path, filetype="pdf")
    else:
        doc = pymupdf.open(pdf_path)
    mat = pymupdf.Matrix(zoom, zoom)
    images = {}
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        jpeg_bytes = pix.tobytes("jpeg")
        images[i + 1] = base64.b64encode(jpeg_bytes).decode("utf-8")
    doc.close()
    return images


def _extract_page_indices_from_prompt(prompt: str, total_pages: int) -> list[int]:
    """
    Best-effort: find physical_index markers like <physical_index_3> in the prompt
    and return the corresponding page numbers.  Falls back to all pages if none found.
    """
    import re
    found = [int(m) for m in re.findall(r"physical_index_(\d+)", prompt)]
    if found:
        return sorted(set(p for p in found if 1 <= p <= total_pages))
    # No markers — return empty list (caller decides whether to include images)
    return []


def _build_vision_messages(prompt: str, page_indices: list[int], page_images: dict) -> list:
    """Build a multimodal messages list with text prompt + page images."""
    content = [{"type": "text", "text": prompt}]
    for idx in page_indices:
        if idx in page_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{page_images[idx]}"},
            })
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def page_index_vision(
    doc,
    vision_model: str,
    vision_api_base: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    toc_model: Optional[str] = None,
    toc_api_base: Optional[str] = None,
    toc_api_key: Optional[str] = None,
    image_zoom: float = 2.0,
    **kwargs,
):
    """
    Drop-in replacement for page_index() that uses vision (multimodal) LLM calls.

    Parameters
    ----------
    doc : str | BytesIO
        Path to a PDF file or a BytesIO object.
    vision_model : str
        LiteLLM model string for the multimodal model, e.g. "openai/gpt-4o".
    vision_api_base : str, optional
        API base URL for the vision model.
    vision_api_key : str, optional
        API key for the vision model.
    toc_model : str, optional
        Text-only model for TOC detection / extraction steps.
        Defaults to vision_model if not set.
    toc_api_base / toc_api_key : str, optional
        Credentials for the toc_model. Defaults to vision_* values.
    image_zoom : float
        Render scale for PDF→image conversion (default 2.0 = 2×).
    **kwargs
        Forwarded to page_index() (model, if_add_node_summary, etc.).
        The `model` kwarg is set to vision_model if not explicitly provided.
    """
    # Resolve model forwarded to page_index()
    if "model" not in kwargs:
        kwargs["model"] = vision_model
    if "api_base" not in kwargs and vision_api_base:
        kwargs["api_base"] = vision_api_base
    if "api_key" not in kwargs and vision_api_key:
        kwargs["api_key"] = vision_api_key

    toc_model = toc_model or vision_model
    toc_api_base = toc_api_base or vision_api_base
    toc_api_key = toc_api_key or vision_api_key

    # 1. Pre-render all PDF pages to images
    print("[vision] Rendering PDF pages to images…")
    page_images = _render_pdf_to_images(doc, zoom=image_zoom)
    total_pages = len(page_images)
    print(f"[vision] Rendered {total_pages} pages.")

    # 2. Import the utils module so we can patch it
    from pageindex import utils as _utils

    # Save originals
    _orig_llm_completion = _utils.llm_completion
    _orig_llm_acompletion = _utils.llm_acompletion

    # 3. Build patched replacements
    def _vision_llm_completion(
        model, prompt, chat_history=None, return_finish_reason=False,
        api_base=None, api_key=None,
    ):
        page_indices = _extract_page_indices_from_prompt(prompt, total_pages)
        if page_indices:
            # Vision call: use vision model + images
            import litellm
            messages = _build_vision_messages(prompt, page_indices, page_images)
            llm_kwargs = _utils._build_llm_kwargs(
                vision_model,
                api_base=vision_api_base,
                api_key=vision_api_key,
            )
            max_retries = 10
            import time
            import logging
            for i in range(max_retries):
                try:
                    response = litellm.completion(
                        **llm_kwargs,
                        messages=messages,
                        temperature=0,
                    )
                    content = response.choices[0].message.content
                    if return_finish_reason:
                        finish_reason = (
                            "max_output_reached"
                            if response.choices[0].finish_reason == "length"
                            else "finished"
                        )
                        return content, finish_reason
                    return content
                except Exception as e:
                    logging.error(f"[vision] llm_completion error: {e}")
                    if i < max_retries - 1:
                        time.sleep(1)
                    else:
                        return ("", "error") if return_finish_reason else ""
        else:
            # No page markers — plain text call with toc_model
            return _orig_llm_completion(
                toc_model, prompt,
                chat_history=chat_history,
                return_finish_reason=return_finish_reason,
                api_base=toc_api_base,
                api_key=toc_api_key,
            )

    async def _vision_llm_acompletion(
        model, prompt, api_base=None, api_key=None,
    ):
        page_indices = _extract_page_indices_from_prompt(prompt, total_pages)
        if page_indices:
            import litellm
            import logging
            messages = _build_vision_messages(prompt, page_indices, page_images)
            llm_kwargs = _utils._build_llm_kwargs(
                vision_model,
                api_base=vision_api_base,
                api_key=vision_api_key,
            )
            max_retries = 10
            for i in range(max_retries):
                try:
                    response = await litellm.acompletion(
                        **llm_kwargs,
                        messages=messages,
                        temperature=0,
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    logging.error(f"[vision] llm_acompletion error: {e}")
                    if i < max_retries - 1:
                        await asyncio.sleep(1)
                    else:
                        return ""
        else:
            return await _orig_llm_acompletion(
                toc_model, prompt,
                api_base=toc_api_base,
                api_key=toc_api_key,
            )

    # 4. Patch
    _utils.llm_completion = _vision_llm_completion
    _utils.llm_acompletion = _vision_llm_acompletion

    try:
        # 5. Run the normal page_index pipeline
        from pageindex.page_index import page_index
        result = page_index(doc, **kwargs)
    finally:
        # 6. Always restore originals
        _utils.llm_completion = _orig_llm_completion
        _utils.llm_acompletion = _orig_llm_acompletion

    return result
