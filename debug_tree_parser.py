"""
调试脚本：直接调用 tree_parser，绕过 FastAPI/BackgroundTasks。
运行方式：
    cd PageIndex
    /opt/homebrew/Caskroom/miniforge/base/envs/rag/bin/python debug_tree_parser.py
"""
import asyncio
import sys
import json
import logging

sys.path.insert(0, ".")

from pageindex.page_index import tree_parser, get_page_tokens
from pageindex.utils import ConfigLoader

# ── 配置 ──────────────────────────────────────────────────────────
PDF_PATH = "/Users/admin/Downloads/【好物众测】商家常见问题QA_商家飞书群整理.pdf"

opt = ConfigLoader().load({
    "model": "openai/deepseek-v3",
    "api_base": "https://dwai.shizhuang-inc.com/ai/gateway",
    "api_key": "dw-PDUDwwoa3sYszEZ4LhjtgUxl2Ptw6e0lMH7lJtb6zbU",
    "if_add_node_id": "yes",
    "if_add_node_text": "yes",
    "if_add_node_summary": "no",        # 调试时关掉，省时间
    "if_add_doc_description": "no",
})
# ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("debug_tree_parser")


async def main():
    print(f"Parsing PDF: {PDF_PATH}")
    page_list = get_page_tokens(PDF_PATH, model=opt.model)
    print(f"Total pages: {len(page_list)}")

    result = await tree_parser(page_list, opt, doc=PDF_PATH, logger=logger)

    print("\n── Result (first 3000 chars) ──")
    print(json.dumps(result, ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    asyncio.run(main())
