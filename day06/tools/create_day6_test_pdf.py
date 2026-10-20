from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


day_directory = Path(__file__).resolve().parent.parent
output_path = day_directory / "documents" / "pdf" / "day6_rag_test.pdf"
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

document = SimpleDocTemplate(
    str(output_path),
    pagesize=A4,
    rightMargin=24 * mm,
    leftMargin=24 * mm,
    topMargin=22 * mm,
    bottomMargin=20 * mm,
    title="RAG 学习资料",
    author="LangChain Day6"
)

styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "ChineseTitle",
    fontName="STSong-Light",
    fontSize=24,
    leading=34,
    textColor=HexColor("#17324D"),
    alignment=TA_LEFT,
    leftIndent=0,
    rightIndent=0,
    firstLineIndent=0,
    wordWrap="CJK",
    spaceAfter=12 * mm
)
heading_style = ParagraphStyle(
    "ChineseHeading",
    parent=styles["Heading2"],
    fontName="STSong-Light",
    fontSize=16,
    leading=24,
    textColor=HexColor("#176B87"),
    spaceBefore=5 * mm,
    spaceAfter=3 * mm
)
body_style = ParagraphStyle(
    "ChineseBody",
    parent=styles["BodyText"],
    fontName="STSong-Light",
    fontSize=11.5,
    leading=20,
    textColor=HexColor("#273444"),
    spaceAfter=4 * mm
)
note_style = ParagraphStyle(
    "ChineseNote",
    parent=body_style,
    backColor=HexColor("#EAF5F7"),
    borderColor=HexColor("#83BBC5"),
    borderWidth=0.8,
    borderPadding=8,
    spaceBefore=5 * mm,
    spaceAfter=5 * mm
)

story = [
    Paragraph("RAG 学习资料", title_style),
    Paragraph("第一部分：什么是 RAG", heading_style),
    Paragraph(
        "RAG 的中文名称是检索增强生成。它让大语言模型在回答问题之前，"
        "先从外部知识库中查找相关资料，再根据检索结果组织答案。",
        body_style
    ),
    Paragraph(
        "RAG 的目标不是重新训练模型，而是在调用模型时为它补充当前问题需要的背景资料。"
        "知识库可以来自文本文件、Markdown、PDF、数据库或者企业内部文档。",
        body_style
    ),
    Paragraph(
        "核心流程：用户问题 → Embedding → 向量检索 → 相关文本块 → 大模型生成回答。",
        note_style
    ),
    Paragraph("第二部分：主要组成", heading_style),
    Paragraph(
        "文档加载器负责把不同格式的文件转换成 Document；文本切分器把长文档切成较小的 chunk；"
        "Embedding 模型把文字转换成向量；向量数据库负责保存和检索向量；聊天模型根据资料回答问题。",
        body_style
    ),
    PageBreak(),
    Paragraph("RAG 实践要点", title_style),
    Paragraph("一、保留来源信息", heading_style),
    Paragraph(
        "每个 Document 都应该保留 source、title、page 等 metadata。这样最终回答可以展示资料来源，"
        "用户也能回到原文核对答案。PDF 通常是一页对应一个 Document。",
        body_style
    ),
    Paragraph("二、控制文本块大小", heading_style),
    Paragraph(
        "文本块太大可能包含过多无关内容，文本块太小则可能破坏完整语义。"
        "chunk_overlap 可以在相邻文本块之间保留少量重叠内容。",
        body_style
    ),
    Paragraph("三、避免重复建立索引", heading_style),
    Paragraph(
        "可以使用文件指纹判断文档是否发生变化。未变化文件跳过加载、切分和 Embedding；"
        "新增文件建立新索引；修改文件替换旧索引；删除文件清理对应的文本块。",
        body_style
    ),
    Paragraph(
        "测试问题建议：什么是 RAG？RAG 包含哪些主要组件？为什么要保留 metadata？",
        note_style
    )
]


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("STSong-Light", 9)
    canvas.setFillColor(HexColor("#6B7785"))
    canvas.drawRightString(A4[0] - 24 * mm, 12 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


document.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number
)

print(output_path)
