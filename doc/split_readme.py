import re
from pathlib import Path

README_FILE = Path("..", "README.rst")
DOCS_DIR = Path(".")
INDEX_RST_FILE = DOCS_DIR / "index.rst"

# Read README
text = README_FILE.read_text(encoding="utf-8")

# Step 1 — Extract intro (title and description)
# Matches the very first section with = underlining (project title)
intro_pattern = re.compile(
    r"^(?P<over>=+)\n(?P<title>[^\n]+)\n(?P=over)\n(?P<body>.*?)(?=\n\S+\n[-=]+\n)", 
    re.DOTALL
)

intro_match = intro_pattern.search(text)

if not intro_match:
    raise ValueError(
        "Could not find the top-level title and intro section in README.rst."
    )

intro_content = intro_match.group(0).strip()
intro_body = intro_match.group("body").strip()

# Write the intro section (title + description) to readme.rst
(DOCS_DIR / "readme.rst").write_text(
    f".. This file is auto-generated. Do not edit.\n\n{intro_content}\n",
    encoding="utf-8"
)

print("Wrote: readme.rst")

# Step 2 — Extract all lower sections (Installation, Usage, etc.)
section_pattern = re.compile(
    r"^(?P<title>[^\n]+)\n(?P<underline>-{3,}|={3,})\n", re.MULTILINE
)
matches = list(section_pattern.finditer(text, intro_match.end()))

# Extract each section by span
section_spans = []
for i, match in enumerate(matches):
    title = match.group("title").strip()
    start = match.start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    section_spans.append((title, text[start:end].strip()))

# Write each section to a separate file
section_filenames = []
for title, content in section_spans:
    filename = title.lower().replace(" ", "_").replace("/", "_")
    section_filenames.append(filename)
    filepath = DOCS_DIR / f"{filename}.rst"

    lines = content.splitlines()
    # Look for heading underline in the second line
    if len(lines) >= 2 and re.match(r"^[=~\-]+$", lines[1]):
        heading = lines[0].strip()
        old_underline = lines[1].strip()

        # Determine new underline based on old one
        if old_underline.startswith("~"):
            new_char = "-"
        elif old_underline.startswith("-"):
            new_char = "="
        else:
            new_char = old_underline[0]  # fallback

        new_underline = new_char * len(heading)
        lines[0:2] = [heading, new_underline]

    upgraded_content = "\n".join(lines).strip()
    upgraded_content = (
        f".. This file is auto-generated. Do not edit."
        f"\n\n{upgraded_content}\n"
    )
    filepath.write_text(upgraded_content, encoding="utf-8")
    print(f"Wrote: {filename}.rst")

# Step 3 — Update index.rst
index_lines = [
    "Welcome to obsweatherscale",
    "=" * 45,
    "",
    ".. toctree::",
    "   :maxdepth: 2",
    "   :caption: Contents:",
    "",
    "   readme",
]

# Add dynamically generated section files (in original README order)
for name in section_filenames:
    index_lines.append(f"   {name}")

# Append static history file manually
index_lines.append("   history")

# Add indices and search
index_lines += [
    "",
    "Indices and tables",
    "==================",
    "* :ref:`genindex`",
    "* :ref:`modindex`",
    "* :ref:`search`",
]

INDEX_RST_FILE.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
print("Updated: index.rst")