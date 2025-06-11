import re
from pathlib import Path

README_FILE = Path("..", "README.rst")
DOCS_DIR = Path(".")
INDEX_RST_FILE = DOCS_DIR / "index.rst"

# Read README
readme_text = README_FILE.read_text(encoding="utf-8")

# Step 1 — Extract intro (title and description)
# Matches the very first section with = underlining (project title)
intro_pattern = re.compile(
    r"^(?P<over>=+)\n(?P<title>[^\n]+)\n(?P=over)\n(?P<body>.*?)(?=\n\S+\n[-=]+\n)", 
    re.DOTALL
)

intro_match = intro_pattern.search(readme_text)

if not intro_match:
    raise ValueError(
        "Could not find the top-level title and intro section in README.rst."
    )

intro_text = intro_match.group(0)
remaining_text = readme_text[intro_match.end():]

# Extract all section blocks: title + underline + content
section_pattern = re.compile(
    r"\n(?P<title>[^\n]+)\n(?P<underline>[-=~]+)\n(?P<body>.*?)(?=\n\S+\n[-=~]+\n|\Z)",
    re.DOTALL,
)

section_matches = list(section_pattern.finditer(remaining_text))

# Prepare groups
special_sections = {"installation", "usage"}
special_filenames = []
other_sections = []

# Process all sections
for match in section_matches:
    title = match.group("title").strip()
    slug = title.lower().replace(" ", "_").replace("/", "_")
    underline = match.group("underline").strip()
    body = match.group("body").strip()
    lines = [title, underline] + body.splitlines()

    if slug in special_sections:
        # Promote heading underline
        if underline.startswith("~"):
            new_char = "-"
        elif underline.startswith("-"):
            new_char = "="
        else:
            new_char = underline[0]

        new_underline = new_char * len(title)
        lines[0:2] = [title, new_underline]

        upgraded_content = f".. This file is auto-generated. Do not edit.\n\n" + "\n".join(lines).strip()
        (DOCS_DIR / f"{slug}.rst").write_text(upgraded_content + "\n", encoding="utf-8")
        special_filenames.append(slug)
        print(f"Wrote: {slug}.rst")
    else:
        # Add to readme.rst later
        other_sections.append("\n".join(lines).strip())

# Combine intro + other sections into readme.rst (no heading promotion)
readme_rst = (
    ".. This file is auto-generated. Do not edit.\n\n" +
    intro_text.strip() + "\n\n" +
    "\n\n".join(other_sections)
)

(DOCS_DIR / "readme.rst").write_text(readme_rst.strip() + "\n", encoding="utf-8")
print("Wrote: readme.rst")