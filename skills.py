"""
Agent Skills integration module.

Supports discovering, loading, and executing external skills/plugins for the Research Agent.
Follows the AgentSkills specification.
"""

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_SCRIPT_TIMEOUT = 60


@dataclass
class SkillMetadata:
    """Represents metadata of a discovered skill."""

    name: str
    description: str
    path: str                      # Absolute path to skill directory
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    allowed_tools: Optional[str] = None


# ====================== Skill Discovery ======================
def parse_skill_frontmatter(skill_md_path: str) -> Optional[SkillMetadata]:
    """Parse YAML frontmatter from SKILL.md file."""
    try:
        with open(skill_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract YAML frontmatter
        match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
        if not match:
            return None

        frontmatter = yaml.safe_load(match.group(1))
        if not frontmatter:
            return None

        name = frontmatter.get("name")
        description = frontmatter.get("description")

        if not name or not description:
            return None

        # Validate skill name format
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", name):
            logger.warning(f"Invalid skill name format: {name}")
            return None

        skill_dir = str(Path(skill_md_path).parent.resolve())

        return SkillMetadata(
            name=name,
            description=description,
            path=skill_dir,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata", {}),
        )

    except Exception as e:
        logger.warning(f"Failed to parse skill at {skill_md_path}: {e}")
        return None


def discover_skills(skill_directories: List[str]) -> List[SkillMetadata]:
    """Discover all skills from given directories."""
    skills = []

    for skill_root in skill_directories:
        skill_path = Path(skill_root)
        if not skill_path.exists() or not skill_path.is_dir():
            logger.warning(f"Skill directory does not exist: {skill_root}")
            continue

        for item in skill_path.iterdir():
            if not item.is_dir():
                continue

            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = parse_skill_frontmatter(str(skill_md))
            if skill:
                skills.append(skill)

    return skills


# ====================== Prompt Generation ======================
def skills_to_xml(skills: List[SkillMetadata]) -> str:
    """Convert skills list to XML format for system prompt."""
    if not skills:
        return ""

    lines = ["<available_skills>"]
    for skill in skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{skill.name}</name>")
        # Escape XML special characters
        escaped_desc = (
            skill.description
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        lines.append(f"    <description>{escaped_desc}</description>")
        skill_md_path = str(Path(skill.path) / "SKILL.md")
        lines.append(f"    <location>{skill_md_path}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")

    return "\n".join(lines)


def build_skills_system_prompt(skills: List[SkillMetadata]) -> str:
    """Build the skills section for system prompt."""
    if not skills:
        return ""

    skills_xml = skills_to_xml(skills)

    return f"""
<agent_skills>
When users ask you to perform tasks, check if any of the available skills below can help complete the task more effectively.

To use a skill:
1. Read the skill's SKILL.md file using the load_skill_file tool.
2. Follow the instructions in the skill documentation.
3. Use execute_script tool when the skill requires running scripts.

Notes:
- Only use skills when they are relevant to the current task.
- Do not load the same file multiple times.
- Skills are not tools themselves — use only the provided tools.

{skills_xml}
</agent_skills>
"""


# ====================== Skill Execution Tools ======================
class SkillIntegrationTools:
    """Tools for loading and executing skills."""

    def __init__(self, skills: List[SkillMetadata]):
        self.skills = {skill.name: skill for skill in skills}

    def load_skill_file(self, skill_name: str, file_path: str = "SKILL.md") -> str:
        """Load a file from a skill directory."""
        try:
            if skill_name not in self.skills:
                return f"Error: Skill '{skill_name}' not found."

            skill_root = Path(self.skills[skill_name].path).resolve()
            target_path = (skill_root / file_path).resolve()

            # Security: prevent path traversal
            if not target_path.is_relative_to(skill_root):
                return f"Error: Access denied. File '{file_path}' is outside the skill directory."

            if not target_path.exists():
                return f"Error: File '{file_path}' not found in skill '{skill_name}'."

            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()

        except Exception as e:
            logger.warning(f"Failed to load file {file_path} from skill {skill_name}: {e}")
            return f"Error: Failed to load file from skill '{skill_name}': {str(e)}"

    def execute_script(self, skill_name: str, command: str) -> str:
        """Execute a script provided by the skill."""
        try:
            if skill_name not in self.skills:
                return f"Error: Skill '{skill_name}' not found."

            skill_root = Path(self.skills[skill_name].path).resolve()

            completed = subprocess.run(
                command,
                shell=True,
                cwd=skill_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=DEFAULT_SCRIPT_TIMEOUT,
                text=True,
                errors="replace",
            )

            stdout_str = (completed.stdout or "").strip()
            stderr_str = (completed.stderr or "").strip()

            logger.info(f"Executed script '{command}' in skill '{skill_name}'")

            return f"<stdout>\n{stdout_str}\n</stdout>\n<stderr>\n{stderr_str}\n</stderr>"

        except subprocess.TimeoutExpired:
            return "<stdout></stdout><stderr>Error: Script execution timed out.</stderr>"
        except Exception as e:
            logger.warning(f"Failed to execute script in skill {skill_name}: {e}")
            return f"<stdout></stdout><stderr>System Error: {str(e)}</stderr>"
