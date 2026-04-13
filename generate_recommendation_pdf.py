from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "recommendation.md"
OUTPUT = ROOT / "recommendation.pdf"


def build_story(markdown_text: str):
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=6,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=14,
        firstLineIndent=-8,
    )
    story = []

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.08 * inch))
            continue

        if line.startswith("# "):
            story.append(Paragraph(line[2:], styles["Title"]))
            story.append(Spacer(1, 0.12 * inch))
            continue

        if line.startswith("### "):
            story.append(Paragraph(line[4:], styles["Heading2"]))
            story.append(Spacer(1, 0.06 * inch))
            continue

        if line.startswith("## "):
            story.append(Paragraph(line[3:], styles["Heading1"]))
            story.append(Spacer(1, 0.08 * inch))
            continue

        if line[:2].isdigit() and line[1:3] == ". ":
            text = f"{line[:1]}. {line[3:]}"
            story.append(Paragraph(text, body))
            continue

        if line.startswith("- "):
            story.append(Paragraph(f"• {line[2:]}", bullet))
            continue

        story.append(Paragraph(line, body))

    return story


def main():
    markdown_text = SOURCE.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=LETTER,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="CV Project Recommendation",
    )
    doc.build(build_story(markdown_text))


if __name__ == "__main__":
    main()
