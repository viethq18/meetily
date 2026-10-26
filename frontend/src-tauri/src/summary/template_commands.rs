use crate::summary::templates;
use serde::{Deserialize, Serialize};
use tauri::Runtime;
use tracing::{info, warn};

/// Template metadata for UI display
#[derive(Debug, Serialize, Deserialize)]
pub struct TemplateInfo {
    /// Template identifier (e.g., "daily_standup", "standard_meeting")
    pub id: String,

    /// Display name for the template
    pub name: String,

    /// Brief description of the template's purpose
    pub description: String,
}

/// Detailed template structure for preview/debugging
#[derive(Debug, Serialize, Deserialize)]
pub struct TemplateDetails {
    /// Template identifier
    pub id: String,

    /// Display name
    pub name: String,

    /// Description
    pub description: String,

    /// List of section titles in order
    pub sections: Vec<String>,
}

/// A template ready for the editor: markdown form plus editability metadata
#[derive(Debug, Serialize, Deserialize)]
pub struct EditableTemplate {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Markdown form (what the editor shows and edits)
    pub markdown: String,
    /// Read-only when the ID is a built-in template with no custom override
    pub is_builtin: bool,
    /// True when a user-writable custom file backs this ID
    pub is_custom: bool,
}

/// Lists all available templates
///
/// Returns templates from both built-in (embedded) and custom (user data directory) sources.
/// Templates are automatically discovered - no code changes needed to add new templates.
///
/// # Returns
/// Vector of TemplateInfo with id, name, and description for each template
#[tauri::command]
pub async fn api_list_templates<R: Runtime>(
    _app: tauri::AppHandle<R>,
) -> Result<Vec<TemplateInfo>, String> {
    info!("api_list_templates called");

    let templates = templates::list_templates();

    let template_infos: Vec<TemplateInfo> = templates
        .into_iter()
        .map(|(id, name, description)| TemplateInfo {
            id,
            name,
            description,
        })
        .collect();

    info!("Found {} available templates", template_infos.len());

    Ok(template_infos)
}

/// Gets detailed information about a specific template
///
/// # Arguments
/// * `template_id` - Template identifier (e.g., "daily_standup")
///
/// # Returns
/// TemplateDetails with full template structure
#[tauri::command]
pub async fn api_get_template_details<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_id: String,
) -> Result<TemplateDetails, String> {
    info!("api_get_template_details called for template_id: {}", template_id);

    let template = templates::get_template(&template_id)?;

    let section_titles: Vec<String> = template
        .sections
        .iter()
        .map(|section| section.title.clone())
        .collect();

    let details = TemplateDetails {
        id: template_id,
        name: template.name,
        description: template.description,
        sections: section_titles,
    };

    info!("Retrieved template details for '{}'", details.name);

    Ok(details)
}

/// Validates a custom template JSON string
///
/// Useful for template editor UI or validation before saving custom templates
///
/// # Arguments
/// * `template_json` - Raw JSON string of the template
///
/// # Returns
/// Ok(template_name) if valid, Err(error_message) if invalid
#[tauri::command]
pub async fn api_validate_template<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_json: String,
) -> Result<String, String> {
    info!("api_validate_template called");

    match templates::validate_and_parse_template(&template_json) {
        Ok(template) => {
            info!("Template '{}' validated successfully", template.name);
            Ok(template.name)
        }
        Err(e) => {
            warn!("Template validation failed: {}", e);
            Err(e)
        }
    }
}

/// Gets a template in editable Markdown form
#[tauri::command]
pub async fn api_get_template_markdown<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_id: String,
) -> Result<EditableTemplate, String> {
    let template = templates::get_template(&template_id)?;
    let is_custom = templates::custom_template_exists(&template_id);
    let is_builtin = templates::is_builtin_template_id(&template_id) && !is_custom;
    Ok(EditableTemplate {
        id: template_id,
        markdown: templates::template_to_markdown(&template),
        name: template.name,
        description: template.description,
        is_builtin,
        is_custom,
    })
}

/// Saves (creates or updates) a custom template from Markdown
///
/// # Arguments
/// * `template_id` - Target ID; empty string for new templates (one is generated)
/// * `markdown` - Meetily template Markdown (see `parse_template_markdown`)
///
/// # Returns
/// The ID of the saved template
#[tauri::command]
pub async fn api_save_template_from_markdown<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_id: String,
    markdown: String,
) -> Result<String, String> {
    let template = templates::parse_template_markdown(&markdown)?;

    let id = if template_id.trim().is_empty() {
        slug_from_name(&template.name)
    } else {
        template_id
    };

    templates::save_custom_template(&id, &template)?;
    info!("Saved template '{}' (name: '{}')", id, template.name);
    Ok(id)
}

/// Deletes a custom template
///
/// Fails when the ID refers to a built-in template without a custom file.
/// Returns Ok(true) when deleted, Ok(false) when no custom file existed.
#[tauri::command]
pub async fn api_delete_template<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_id: String,
) -> Result<bool, String> {
    if templates::is_builtin_template_id(&template_id) && !templates::custom_template_exists(&template_id) {
        return Err(format!("Built-in template '{}' cannot be deleted", template_id));
    }
    templates::delete_custom_template(&template_id)
}

/// Exports a template as a Meetily Markdown string
///
/// The frontend writes this to a user-chosen `.md` file.
#[tauri::command]
pub async fn api_export_template_markdown<R: Runtime>(
    _app: tauri::AppHandle<R>,
    template_id: String,
) -> Result<String, String> {
    let template = templates::get_template(&template_id)?;
    Ok(templates::template_to_markdown(&template))
}

/// Imports a Meetily template Markdown string and saves it as a custom template
///
/// # Arguments
/// * `markdown` - File content previously exported or hand-written
///
/// # Returns
/// The ID of the imported template
#[tauri::command]
pub async fn api_import_template_markdown<R: Runtime>(
    _app: tauri::AppHandle<R>,
    markdown: String,
) -> Result<String, String> {
    let template = templates::parse_template_markdown(&markdown)?;
    let id = slug_from_name(&template.name);
    templates::save_custom_template(&id, &template)?;
    info!("Imported template '{}' (name: '{}')", id, template.name);
    Ok(id)
}

/// Derive a filesystem-safe template ID from a display name
fn slug_from_name(name: &str) -> String {
    let slug: String = name
        .to_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c
            } else if c.is_whitespace() {
                '_'
            } else {
                '-'
            }
        })
        .collect();
    let trimmed = slug.trim_matches(|c: char| !c.is_ascii_alphanumeric());
    let base = if trimmed.is_empty() { "custom_template".to_string() } else { trimmed.to_string() };

    // Avoid silently shadowing a built-in: append a suffix on collision
    let mut id = base.clone();
    let mut n = 2;
    while templates::is_builtin_template_id(&id) {
        id = format!("{}_{}", base, n);
        n += 1;
    }
    id
}

#[cfg(test)]
mod slug_tests {
    use super::*;

    #[test]
    fn test_slug_from_name_basic() {
        assert_eq!(slug_from_name("Client Call Notes"), "client_call_notes");
    }

    #[test]
    fn test_slug_from_name_builtin_collision() {
        // "Standard Meeting Notes" would slug to standard_meeting_notes; use the exact name
        let id = slug_from_name("Daily Standup");
        assert_ne!(id, "daily_standup");
        assert_eq!(id, "daily_standup_2");
    }

    #[test]
    fn test_slug_from_name_empty() {
        assert_eq!(slug_from_name("###"), "custom_template");
    }
}

#[cfg(test)]
mod markdown_tests {
    use crate::summary::templates;

    #[test]
    fn test_markdown_round_trip_builtin() {
        let template = templates::get_template("standard_meeting").expect("builtin exists");
        let markdown = templates::template_to_markdown(&template);
        let parsed = templates::parse_template_markdown(&markdown).expect("round trip parses");

        assert_eq!(parsed.name, template.name);
        assert_eq!(parsed.description.trim(), template.description.trim());
        assert_eq!(parsed.sections.len(), template.sections.len());
        for (a, b) in parsed.sections.iter().zip(template.sections.iter()) {
            assert_eq!(a.title, b.title);
            assert_eq!(a.instruction, b.instruction);
            assert_eq!(a.item_format, b.item_format);
        }
    }

    #[test]
    fn test_parse_markdown_missing_title() {
        let result = templates::parse_template_markdown("## Section\n\nInstruction");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_markdown_no_sections() {
        let result = templates::parse_template_markdown("# Title\n\nDescription only");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_markdown_table_item_format() {
        let md = "# T\n\nD\n\n## Actions\n\nList tasks.\n\n| **Owner** | Task |\n| --- | --- |\n";
        let parsed = templates::parse_template_markdown(md).expect("parses");
        assert_eq!(parsed.sections.len(), 1);
        assert!(parsed.sections[0].item_format.as_deref().unwrap().starts_with("| **Owner**"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_templates() {
        // This test requires the templates to be embedded/available
        // In a real test environment, you might want to mock the templates module

        // For now, just verify the function compiles and runs
        // You can expand this with more specific assertions
    }

    #[tokio::test]
    async fn test_validate_template_valid() {
        let valid_json = r#"
        {
            "name": "Test Template",
            "description": "A test template",
            "sections": [
                {
                    "title": "Summary",
                    "instruction": "Provide a summary",
                    "format": "paragraph"
                }
            ]
        }"#;

        // Mock app handle would be needed for actual testing
        // For now, test the validation logic directly
        let result = templates::validate_and_parse_template(valid_json);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_template_invalid() {
        let invalid_json = "invalid json";

        let result = templates::validate_and_parse_template(invalid_json);
        assert!(result.is_err());
    }
}
