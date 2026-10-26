"use client";

import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { toast } from "sonner";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { Plus, Copy, Pencil, Trash2, Download, Upload } from "lucide-react";

interface TemplateInfo {
    id: string;
    name: string;
    description: string;
}

interface EditableTemplate {
    id: string;
    name: string;
    description: string;
    markdown: string;
    is_builtin: boolean;
    is_custom: boolean;
}

type EditorState =
    | { kind: 'closed' }
    | { kind: 'new' }
    | { kind: 'edit'; templateId: string; builtin: boolean };

const TEMPLATE_MD_HINT = `# Template Title

One-line description of what this template captures.

## Section Name

Instruction for the AI on what to extract in this section.

Optional table hint (lines starting with | are kept as the item format):
| **Owner** | Task |
| --- | --- |`;

/**
 * Summary template management: list, create, edit, copy, delete, import, export.
 *
 * Templates are edited as Markdown and persisted as JSON by the Rust core
 * (`api_save_template_from_markdown`). Built-in templates are read-only;
 * copying one creates an editable custom template.
 */
export function TemplateSettings() {
    const [templates, setTemplates] = useState<TemplateInfo[]>([]);
    const [selectedId, setSelectedId] = useState<string>('standard_meeting');
    const [loading, setLoading] = useState(true);
    const [editor, setEditor] = useState<EditorState>({ kind: 'closed' });
    const [editorMarkdown, setEditorMarkdown] = useState('');
    const [saving, setSaving] = useState(false);
    const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null);

    const fetchTemplates = useCallback(async () => {
        try {
            const list = await invoke<TemplateInfo[]>('api_list_templates');
            setTemplates(list);
        } catch (err) {
            console.error('Failed to fetch templates:', err);
            toast.error('Failed to load templates', {
                description: err instanceof Error ? err.message : String(err),
            });
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchTemplates();
        const stored = typeof window !== 'undefined' ? window.localStorage.getItem('meetily_selected_template') : null;
        if (stored) setSelectedId(stored);
    }, [fetchTemplates]);

    const selectTemplate = (id: string) => {
        setSelectedId(id);
        window.localStorage.setItem('meetily_selected_template', id);
    };

    const openNew = () => {
        setEditorMarkdown(TEMPLATE_MD_HINT);
        setEditor({ kind: 'new' });
    };

    const openEdit = async (templateId: string) => {
        try {
            const t = await invoke<EditableTemplate>('api_get_template_markdown', { templateId });
            setEditorMarkdown(t.markdown);
            setEditor({ kind: 'edit', templateId, builtin: t.is_builtin });
        } catch (err) {
            toast.error('Failed to open template', {
                description: err instanceof Error ? err.message : String(err),
            });
        }
    };

    // Copy a built-in (or custom) template into a new editable draft
    const copyTemplate = async (templateId: string) => {
        try {
            const t = await invoke<EditableTemplate>('api_get_template_markdown', { templateId });
            // Force a distinct title so save generates a new template
            const renamed = t.markdown.replace(/^# (.+)$/m, `# ${t.name} (Copy)`);
            setEditorMarkdown(renamed);
            setEditor({ kind: 'new' });
        } catch (err) {
            toast.error('Failed to copy template', {
                description: err instanceof Error ? err.message : String(err),
            });
        }
    };

    const saveEditor = async () => {
        if (saving) return;
        setSaving(true);
        try {
            const templateId = editor.kind === 'edit' ? editor.templateId : '';
            const savedId = await invoke<string>('api_save_template_from_markdown', {
                templateId,
                markdown: editorMarkdown,
            });
            toast.success('Template saved', { description: `ID: ${savedId}` });
            setEditor({ kind: 'closed' });
            await fetchTemplates();
            selectTemplate(savedId);
        } catch (err) {
            toast.error('Invalid template', {
                description: err instanceof Error ? err.message : String(err),
                duration: 8000,
            });
        } finally {
            setSaving(false);
        }
    };

    const deleteTemplate = async (templateId: string) => {
        try {
            await invoke('api_delete_template', { templateId });
            toast.success('Template deleted');
            setConfirmDeleteId(null);
            if (selectedId === templateId) selectTemplate('standard_meeting');
            await fetchTemplates();
        } catch (err) {
            toast.error('Failed to delete template', {
                description: err instanceof Error ? err.message : String(err),
            });
        }
    };

    const importTemplate = async () => {
        try {
            const markdown = window.prompt('Paste a Meetily template Markdown file content:');
            if (!markdown) return;
            const savedId = await invoke<string>('api_import_template_markdown', { markdown });
            toast.success('Template imported', { description: `ID: ${savedId}` });
            await fetchTemplates();
            selectTemplate(savedId);
        } catch (err) {
            toast.error('Import failed', {
                description: err instanceof Error ? err.message : String(err),
                duration: 8000,
            });
        }
    };

    const exportTemplate = async (templateId: string) => {
        try {
            const markdown = await invoke<string>('api_export_template_markdown', { templateId });
            const blob = new Blob([markdown], { type: 'text/markdown' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${templateId}.md`;
            a.click();
            URL.revokeObjectURL(url);
            toast.success('Template exported', { description: `${templateId}.md` });
        } catch (err) {
            toast.error('Export failed', {
                description: err instanceof Error ? err.message : String(err),
            });
        }
    };

    if (editor.kind !== 'closed') {
        const isBuiltinEdit = editor.kind === 'edit' && editor.builtin;
        return (
            <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                            {editor.kind === 'new' ? 'New Template' : 'Edit Template'}
                        </h3>
                        {editor.kind === 'edit' && (
                            <p className="text-sm text-gray-600 mt-1">{editor.templateId}</p>
                        )}
                    </div>
                </div>

                {isBuiltinEdit && (
                    <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded text-amber-800 text-xs">
                        This is a built-in template. Saving creates a custom override with the same ID.
                    </div>
                )}

                <Textarea
                    value={editorMarkdown}
                    onChange={(e) => setEditorMarkdown(e.target.value)}
                    rows={16}
                    className="font-mono text-xs"
                    spellCheck={false}
                />

                <div className="flex justify-end gap-2 mt-4">
                    <Button variant="outline" onClick={() => setEditor({ kind: 'closed' })}>
                        Cancel
                    </Button>
                    <Button onClick={saveEditor} disabled={saving}>
                        {saving ? 'Saving...' : 'Create Template'}
                    </Button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-4">
            {/* Toolbar card */}
            <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-semibold text-gray-900 mb-2">Summary Templates</h3>
                        <p className="text-sm text-gray-600">
                            The selected template controls the structure of generated summaries.
                            Built-in templates are read-only — copy one to customize.
                        </p>
                    </div>
                    <div className="flex gap-2 shrink-0">
                        <Button variant="outline" size="sm" onClick={importTemplate}>
                            <Upload className="h-4 w-4 mr-1" />
                            Import
                        </Button>
                        <Button size="sm" onClick={openNew}>
                            <Plus className="h-4 w-4 mr-1" />
                            New
                        </Button>
                    </div>
                </div>
            </div>

            {confirmDeleteId && (
                <div className="bg-white rounded-lg border border-red-200 p-4 shadow-sm">
                    <p className="text-sm font-medium text-red-800">Delete this template permanently?</p>
                    <p className="text-xs text-red-600 mt-1">This cannot be undone.</p>
                    <div className="flex gap-2 mt-3">
                        <Button variant="destructive" size="sm" onClick={() => deleteTemplate(confirmDeleteId)}>
                            Delete
                        </Button>
                        <Button variant="outline" size="sm" onClick={() => setConfirmDeleteId(null)}>
                            Cancel
                        </Button>
                    </div>
                </div>
            )}

            {loading ? (
                <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                    <p className="text-sm text-gray-500">Loading templates...</p>
                </div>
            ) : (
                <div className="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {templates.map((t) => {
                            const isBuiltin = t.id === 'daily_standup' || t.id === 'standard_meeting';
                            const isSelected = selectedId === t.id;
                            return (
                                <div
                                    key={t.id}
                                    onClick={() => selectTemplate(t.id)}
                                    className={`border rounded-lg p-4 cursor-pointer transition ${isSelected ? 'border-blue-500 ring-1 ring-blue-500 bg-blue-50/50' : 'border-gray-200 hover:border-gray-300'}`}
                                >
                                    <div className="flex items-start justify-between gap-2">
                                        <div className="min-w-0">
                                            <p className="font-medium text-sm text-gray-900 truncate">
                                                {t.name}
                                            </p>
                                            <p className="text-xs text-gray-500 line-clamp-2 mt-1">{t.description}</p>
                                            <p className="text-[10px] text-gray-400 mt-1.5">
                                                {t.id} {isBuiltin ? '· built-in' : '· custom'}
                                            </p>
                                        </div>
                                        {isSelected && (
                                            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-medium bg-blue-100 text-blue-700 shrink-0">
                                                Active
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex gap-1 mt-3" onClick={(e) => e.stopPropagation()}>
                                        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={() => openEdit(t.id)}>
                                            <Pencil className="h-3 w-3 mr-1" />
                                            Edit
                                        </Button>
                                        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={() => copyTemplate(t.id)}>
                                            <Copy className="h-3 w-3 mr-1" />
                                            Copy
                                        </Button>
                                        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={() => exportTemplate(t.id)}>
                                            <Download className="h-3 w-3 mr-1" />
                                            Export
                                        </Button>
                                        {!isBuiltin && (
                                            <Button variant="ghost" size="sm" className="h-7 px-2 text-xs text-red-600 hover:text-red-700" onClick={() => setConfirmDeleteId(t.id)}>
                                                <Trash2 className="h-3 w-3 mr-1" />
                                                Delete
                                            </Button>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
