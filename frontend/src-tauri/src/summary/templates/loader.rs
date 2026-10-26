use super::defaults;
use super::types::Template;
use std::path::PathBuf;
use tracing::{debug, info, warn};
use once_cell::sync::Lazy;
use std::sync::RwLock;

// Global storage for the bundled templates directory path
static BUNDLED_TEMPLATES_DIR: Lazy<RwLock<Option<PathBuf>>> = Lazy::new(|| RwLock::new(None));

/// Set the bundled templates directory path (called once at app startup)
pub fn set_bundled_templates_dir(path: PathBuf) {
    info!("Bundled templates directory set to: {:?}", path);
    if let Ok(mut dir) = BUNDLED_TEMPLATES_DIR.write() {
        *dir = Some(path);
    }
}

/// Get the user's custom templates directory path
///
/// Returns the platform-specific application data directory for custom templates:
/// - macOS: ~/Library/Application Support/Meetily/templates/
/// - Windows: %APPDATA%\Meetily\templates\
/// - Linux: ~/.config/Meetily/templates/
fn get_custom_templates_dir() -> Option<PathBuf> {
    let mut path = dirs::data_dir()?;
    path.push("Meetily");
    path.push("templates");
    Some(path)
}

/// Load a template from the bundled resources directory
///
/// # Arguments
/// * `template_id` - Template identifier (without .json extension)
///
/// # Returns
/// The template JSON content if found, None otherwise
fn load_bundled_template(template_id: &str) -> Option<String> {
    let bundled_dir = BUNDLED_TEMPLATES_DIR.read().ok()?.clone()?;
    let template_path = bundled_dir.join(format!("{}.json", template_id));

    debug!("Checking for bundled template at: {:?}", template_path);

    match std::fs::read_to_string(&template_path) {
        Ok(content) => {
            info!("Loaded bundled template '{}' from {:?}", template_id, template_path);
            Some(content)
        }
        Err(e) => {
            debug!("No bundled template '{}' found: {}", template_id, e);
            None
        }
    }
}

/// Load a template from the user's custom templates directory
///
/// # Arguments
/// * `template_id` - Template identifier (without .json extension)
///
/// # Returns
/// The template JSON content if found, None otherwise
fn load_custom_template(template_id: &str) -> Option<String> {
    let custom_dir = get_custom_templates_dir()?;
    let template_path = custom_dir.join(format!("{}.json", template_id));

    debug!("Checking for custom template at: {:?}", template_path);

    match std::fs::read_to_string(&template_path) {
        Ok(content) => {
            info!("Loaded custom template '{}' from {:?}", template_id, template_path);
            Some(content)
        }
        Err(e) => {
            debug!("No custom template '{}' found: {}", template_id, e);
            None
        }
    }
}

/// Load and parse a template by identifier
///
/// This function implements a fallback strategy:
/// 1. Check user's custom templates directory
/// 2. Check bundled resources directory (app templates)
/// 3. Fall back to built-in embedded templates
/// 4. Return error if not found in any location
///
/// # Arguments
/// * `template_id` - Template identifier (e.g., "daily_standup", "standard_meeting")
///
/// # Returns
/// Parsed and validated Template struct
pub fn get_template(template_id: &str) -> Result<Template, String> {
    info!("Loading template: {}", template_id);

    // Try custom template first, then bundled, then built-in
    let json_content = if let Some(custom_content) = load_custom_template(template_id) {
        debug!("Using custom template for '{}'", template_id);
        custom_content
    } else if let Some(bundled_content) = load_bundled_template(template_id) {
        debug!("Using bundled template for '{}'", template_id);
        bundled_content
    } else if let Some(builtin_content) = defaults::get_builtin_template(template_id) {
        debug!("Using built-in template for '{}'", template_id);
        builtin_content.to_string()
    } else {
        return Err(format!(
            "Template '{}' not found. Available templates: {}",
            template_id,
            list_template_ids().join(", ")
        ));
    };

    // Parse and validate
    validate_and_parse_template(&json_content)
}

/// Validate and parse template JSON
///
/// # Arguments
/// * `json_content` - Raw JSON string
///
/// # Returns
/// Parsed and validated Template struct
pub fn validate_and_parse_template(json_content: &str) -> Result<Template, String> {
    let template: Template = serde_json::from_str(json_content)
        .map_err(|e| format!("Failed to parse template JSON: {}", e))?;

    template.validate()?;

    Ok(template)
}

/// List all available template identifiers
///
/// Returns a combined list of:
/// - Built-in template IDs
/// - Bundled template IDs (from app resources)
/// - Custom template IDs (from user's data directory)
pub fn list_template_ids() -> Vec<String> {
    let mut ids: Vec<String> = defaults::list_builtin_template_ids()
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    // Add bundled templates if directory is set
    if let Ok(bundled_dir_lock) = BUNDLED_TEMPLATES_DIR.read() {
        if let Some(bundled_dir) = bundled_dir_lock.as_ref() {
            if bundled_dir.exists() {
                match std::fs::read_dir(bundled_dir) {
                    Ok(entries) => {
                        for entry in entries.flatten() {
                            if let Some(filename) = entry.file_name().to_str() {
                                if filename.ends_with(".json") {
                                    let id = filename.trim_end_matches(".json").to_string();
                                    if !ids.contains(&id) {
                                        ids.push(id);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to read bundled templates directory: {}", e);
                    }
                }
            }
        }
    }

    // Add custom templates if directory exists
    if let Some(custom_dir) = get_custom_templates_dir() {
        if custom_dir.exists() {
            match std::fs::read_dir(&custom_dir) {
                Ok(entries) => {
                    for entry in entries.flatten() {
                        if let Some(filename) = entry.file_name().to_str() {
                            if filename.ends_with(".json") {
                                let id = filename.trim_end_matches(".json").to_string();
                                if !ids.contains(&id) {
                                    ids.push(id);
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read custom templates directory: {}", e);
                }
            }
        }
    }

    ids.sort();
    ids
}

/// List all available templates with their metadata
///
/// Returns a list of (id, name, description) tuples
pub fn list_templates() -> Vec<(String, String, String)> {
    let mut templates = Vec::new();

    for id in list_template_ids() {
        match get_template(&id) {
            Ok(template) => {
                templates.push((id, template.name, template.description));
            }
            Err(e) => {
                warn!("Failed to load template '{}': {}", id, e);
            }
        }
    }

    templates
}

/// Save a custom template to the user's custom templates directory
///
/// Creates the directory if it does not exist. Custom templates override
/// bundled/built-in templates with the same ID.
///
/// # Arguments
/// * `template_id` - Template identifier (used as filename stem)
/// * `template` - Validated template to persist
///
/// # Returns
/// The path the template was written to
pub fn save_custom_template(template_id: &str, template: &Template) -> Result<PathBuf, String> {
    template.validate()?;

    if template_id.is_empty()
        || !template_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "Invalid template id '{}': use letters, digits, '_' or '-' only",
            template_id
        ));
    }

    let dir = get_custom_templates_dir()
        .ok_or_else(|| "Could not resolve custom templates directory".to_string())?;
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("Failed to create templates directory {:?}: {}", dir, e))?;

    let path = dir.join(format!("{}.json", template_id));
    let json = serde_json::to_string_pretty(template)
        .map_err(|e| format!("Failed to serialize template: {}", e))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Failed to write template {:?}: {}", path, e))?;

    info!("Saved custom template '{}' to {:?}", template_id, path);
    Ok(path)
}

/// Delete a custom template from the user's custom templates directory
///
/// Returns Ok(false) when no custom template with this ID exists.
pub fn delete_custom_template(template_id: &str) -> Result<bool, String> {
    let dir = get_custom_templates_dir()
        .ok_or_else(|| "Could not resolve custom templates directory".to_string())?;
    let path = dir.join(format!("{}.json", template_id));

    if !path.exists() {
        return Ok(false);
    }

    std::fs::remove_file(&path)
        .map_err(|e| format!("Failed to delete template {:?}: {}", path, e))?;
    info!("Deleted custom template '{}'", template_id);
    Ok(true)
}

/// Whether a template ID is a read-only built-in (embedded) template
pub fn is_builtin_template_id(template_id: &str) -> bool {
    defaults::get_builtin_template(template_id).is_some()
}

/// Whether a custom (user-writable) template with this ID exists
pub fn custom_template_exists(template_id: &str) -> bool {
    get_custom_templates_dir()
        .map(|dir| dir.join(format!("{}.json", template_id)).exists())
        .unwrap_or(false)
}

/// Serialize a template to the Meetily Markdown format
///
/// Format: `# Title`, description paragraph, then for each section a
/// `## Section Title` heading followed by the section instruction (and
/// item_format table line when present). Round-trips with `parse_template_markdown`.
pub fn template_to_markdown(template: &Template) -> String {
    let mut md = format!("# {}\n\n{}\n", template.name, template.description);
    for section in &template.sections {
        md.push_str(&format!("\n## {}\n\n{}\n", section.title, section.instruction));
        if let Some(item_format) = &section.item_format {
            md.push_str(&format!("\n{}\n", item_format));
        }
    }
    md
}

/// Parse the Meetily Markdown template format into a Template
///
/// The first `# Heading` is the template name; text between it and the first
/// `## Heading` is the description; each `## Heading` starts a section whose
/// body (until the next heading) is the instruction. A body line starting with
/// `|` is captured as the section's `item_format` table.
pub fn parse_template_markdown(markdown: &str) -> Result<Template, String> {
    let mut name: Option<String> = None;
    let mut description_lines: Vec<String> = Vec::new();
    // (title, instruction_lines)
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();

    for line in markdown.lines() {
        let trimmed = line.trim_end();
        if let Some(title) = trimmed.strip_prefix("# ") {
            if name.is_none() && sections.is_empty() {
                name = Some(title.trim().to_string());
            } else {
                return Err("Template must contain only one top-level '# Title' heading".to_string());
            }
        } else if let Some(title) = trimmed.strip_prefix("## ") {
            sections.push((title.trim().to_string(), Vec::new()));
        } else if !sections.is_empty() {
            sections
                .last_mut()
                .expect("non-empty sections checked above")
                .1
                .push(trimmed.to_string());
        } else if !trimmed.trim().is_empty() {
            description_lines.push(trimmed.to_string());
        }
    }

    let name = name.ok_or_else(|| "Template must start with a '# Title' heading".to_string())?;
    if sections.is_empty() {
        return Err("Template must include at least one '## Section' heading".to_string());
    }

    let mut parsed_sections = Vec::with_capacity(sections.len());
    for (title, body_lines) in sections {
        let instruction: String = body_lines
            .iter()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('|'))
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string();
        let table_lines: Vec<&String> = body_lines
            .iter()
            .filter(|l| l.trim_start().starts_with('|'))
            .collect();
        // Collapse table lines back into one item_format (they may span multiple lines)
        let item_format = if table_lines.is_empty() {
            None
        } else {
            Some(
                table_lines
                    .iter()
                    .map(|l| l.trim().to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
        };

        parsed_sections.push(super::types::TemplateSection {
            title,
            instruction,
            format: if item_format.is_some() { "list".to_string() } else { "paragraph".to_string() },
            item_format,
            example_item_format: None,
        });
    }

    let template = Template {
        name,
        description: description_lines.join(" ").trim().to_string(),
        sections: parsed_sections,
    };
    template.validate()?;
    Ok(template)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_builtin_template() {
        let template = get_template("daily_standup");
        assert!(template.is_ok());

        let template = template.unwrap();
        assert_eq!(template.name, "Daily Standup");
        assert!(!template.sections.is_empty());
    }

    #[test]
    fn test_get_nonexistent_template() {
        let result = get_template("nonexistent_template");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_template_ids() {
        let ids = list_template_ids();
        assert!(ids.contains(&"daily_standup".to_string()));
        assert!(ids.contains(&"standard_meeting".to_string()));
    }

    #[test]
    fn test_validate_invalid_json() {
        let result = validate_and_parse_template("invalid json");
        assert!(result.is_err());
    }
}
