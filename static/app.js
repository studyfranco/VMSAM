// --- Utilities ---

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' :
        type === 'success' ? 'toast-success' :
            '' // Default is info
        }`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Animate in
    requestAnimationFrame(() => {
        toast.classList.add('visible');
    });

    setTimeout(() => {
        toast.classList.remove('visible');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function copyPattern(pattern) {
    navigator.clipboard.writeText(pattern);
    showToast('Pattern copied to clipboard!', 'success');
}

function toggleHelp() {
    const content = document.getElementById('help-content');
    const icon = document.getElementById('help-icon');
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.textContent = '▲';
    } else {
        content.classList.add('hidden');
        icon.textContent = '▼';
    }
}

function toggleAdvanced() {
    const content = document.getElementById('advanced-options');
    const icon = document.getElementById('adv-icon');
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.textContent = '▼';
    } else {
        content.classList.add('hidden');
        icon.textContent = '▶';
    }
}

// State
let state = {
    currentBaseDir: null,
    selectedFolderId: null,
    selectedFolderPath: null,
    regexCards: [], // { id, filename, regex, rename, weight, valid }
    files: []
};

// --- API Client ---

const api = {
    async listFiles(path, rootType = 'files') {
        const p = path ? `?path=${encodeURIComponent(path)}&root_type=${rootType}` : `?root_type=${rootType}`;
        const res = await fetch(`/api/fs/list${p}`);
        if (!res.ok) throw new Error('Failed to list files');
        return res.json();
    },
    async getFolders() {
        const res = await fetch('/api/vmsam/folders_list');
        if (!res.ok) throw new Error('Failed to fetch folders');
        return res.json();
    },
    async createFolder(payload) {
        const res = await fetch('/api/vmsam/folders', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to create folder');
        return res.json();
    },
    async createRegex(payload) {
        const res = await fetch('/api/vmsam/regex', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Failed to create regex');
        return res.json();
    },
    async getRegexes(folderId) {
        // We assume the backend proxy /api/vmsam/regex_list returns all or filters.
        // If the upstream api supports filtering by folder, we would pass it here.
        // Update: Upstream expects folder_id
        const res = await fetch(`/api/vmsam/regex_list?folder_id=${folderId}`);
        if (!res.ok) throw new Error('Failed to fetch regexes');
        return res.json();
    }
};

// --- UI Logic: Tabs ---

function switchTab(tabId) {
    document.querySelectorAll('main > section').forEach(el => el.classList.add('hidden'));
    document.getElementById(tabId).classList.remove('hidden');

    // Lazy load logic
    if (tabId === 'folder-tab') {
        const container = document.getElementById('dir-browser');
        // Always try to load if empty or stuck loading, or just force reload for freshness
        if (!container.children.length || container.textContent.includes('Loading')) {
            console.log('Switching to folder-tab: Loading directories...');
            loadDirBrowser();
        }
    } else if (tabId === 'regex-tab') {
        console.log('Switching to regex-tab: Initializing...');
        initRegexTab();
    }
}

// --- UI Logic: Folder Creation ---

async function loadDirBrowser(path = '') {
    console.log('loadDirBrowser called with path:', path);
    const container = document.getElementById('dir-browser');
    container.innerHTML = '<div class="loading-message" style="padding:1rem; text-align:center;">Loading...</div>';

    try {
        const items = await api.listFiles(path, 'create');
        container.innerHTML = '';

        // Back button: navigate UP one level
        if (path) {
            // Path is now relative, e.g. "subdir/nested"
            // Split by separator
            const parts = path.split('/');
            const parentPath = parts.slice(0, -1).join('/');

            const backBtn = document.createElement('div');
            backBtn.className = 'dir-item text-blue';
            backBtn.innerHTML = '<span>📁 ..</span>';
            backBtn.onclick = () => loadDirBrowser(parentPath);
            container.appendChild(backBtn);
        }

        if (items.length === 0) {
            container.innerHTML += '<div class="text-muted text-center p-2">No folders found</div>';
        }

        // Render directories
        items.filter(i => i.is_dir).forEach(item => {
            const el = document.createElement('div');
            // Base classes
            const isSelected = state.currentBaseDir === item.path;
            el.className = `dir-item ${isSelected ? 'selected' : ''}`;

            // Row click creates selection
            el.onclick = (e) => selectDir(item.path, e);
            // Double click opens the directory
            el.ondblclick = () => loadDirBrowser(item.path);

            el.innerHTML = `
                <div class="dir-name flex-1">
                    <span>📁 ${item.name}</span>
                </div>
                <div class="dir-actions" style="opacity: 0.8;">
                    <button class="btn btn-sm btn-secondary" onclick="selectDir('${item.path}', event)">Select</button>
                    <button class="btn btn-sm btn-secondary" onclick="event.stopPropagation(); loadDirBrowser('${item.path}')">Open →</button>
                </div>
            `;
            container.appendChild(el);
        });
    } catch (e) {
        console.error('Directory loading error:', e);
        container.innerHTML = `
            <div class="text-accent text-sm p-4 text-center">
                Error: ${e.message} <br>
                <button class="btn btn-sm btn-primary mt-2" onclick="loadDirBrowser('${path}')">Retry</button>
            </div>`;
    }
}

function selectDir(path, event) {
    event.stopPropagation();
    state.currentBaseDir = path;
    updatePreview();

    // Visual feedback
    const container = document.getElementById('dir-browser');
    Array.from(container.children).forEach(child => {
        child.classList.remove('selected');
    });

    // Find the row that contains the button or is the row
    const row = event.target.closest('.dir-item');
    if (row) row.classList.add('selected');

}

function updatePreview() {
    const series = document.getElementById('series-name').value.trim() || '{SeriesName}';
    const tvdb = document.getElementById('tvdb-id').value.trim() || '{TVDB_ID}';
    const sub = document.getElementById('subfolder').value.trim();
    const base = state.currentBaseDir || '[Select Base Dir]';

    const fullPath = `${base}/${series} {tvdb-${tvdb}} [tvdb-${tvdb}] [tvdbid-${tvdb}]/${sub}`;
    document.getElementById('path-preview').textContent = fullPath;
}

async function submitFolder() {
    const seriesName = document.getElementById('series-name').value.trim();
    const tvdbId = document.getElementById('tvdb-id').value.trim();
    const subfolder = document.getElementById('subfolder').value.trim();

    if (!state.currentBaseDir || !seriesName || !tvdbId) {
        showToast('Please fill all fields and select a base directory.', 'error');
        return;
    }

    // Construct destination_path logic
    // Format: {baseDir}/tv-shows/{SeriesName} {tvdb-{TVDB_ID}}/{Subfolder}
    const destinationPath = `${state.currentBaseDir}/${seriesName} {tvdb-${tvdbId}} [tvdb-${tvdbId}] [tvdbid-${tvdbId}]/${subfolder}`;

    // Helper safely get int/float
    const parseIntSafe = (id, def) => {
        const v = parseInt(document.getElementById(id).value, 10);
        return isNaN(v) ? def : v;
    };
    const parseFloatSafe = (id, def) => {
        const v = parseFloat(document.getElementById(id).value);
        return isNaN(v) ? def : v;
    };

    const payload = {
        destination_path: destinationPath,
        original_language: document.getElementById('orig-lang').value.trim() || "en",
        number_cut: parseIntSafe('number-cut', 10),
        cut_file_to_get_delay_second_method: parseFloatSafe('cut-delay', 2.0),
        max_episode_number: parseIntSafe('max-ep', 12)
    };

    try {
        await api.createFolder(payload);
        showToast('Folder created successfully!', 'success');
    } catch (e) {
        console.error('Create folder error:', e);
        showToast('Error creating folder: ' + e.message, 'error');
    }
}

// Listeners
document.getElementById('series-name').addEventListener('input', updatePreview);
document.getElementById('tvdb-id').addEventListener('input', updatePreview);
document.getElementById('subfolder').addEventListener('input', updatePreview);


// --- UI Logic: Regex ---

async function initRegexTab() {
    const input = document.getElementById('folder-search');
    const list = document.getElementById('folder-list');

    input.addEventListener('focus', () => list.classList.remove('hidden'));
    // Close when clicking outside - handled by body click? simpler for now

    let folders = [];
    try {
        console.log('Fetching folders for Regex tab...');
        const response = await api.getFolders();
        // Check if response is array or object with folders property
        if (Array.isArray(response)) {
            folders = response;
        } else if (response.folders && Array.isArray(response.folders)) {
            folders = response.folders;
        } else {
            console.warn('Unexpected folder response structure:', response);
            showToast('Received unexpected data format for folders', 'error');
        }

        if (!folders || folders.length === 0) {
            console.log('No folders returned from API');
            showToast('No folders found to manage', 'info');
        }
    } catch (e) {
        console.error('Error fetching folders:', e);
        showToast('Failed to load folders: ' + e.message, 'error');
    }

    const renderFolders = (query) => {
        list.innerHTML = '';
        const filtered = folders.filter(f => f.destination_path.toLowerCase().includes(query.toLowerCase()));
        filtered.forEach(f => {
            const item = document.createElement('div');
            item.className = 'dir-item text-muted';
            item.textContent = f.destination_path;
            item.onclick = () => {
                state.selectedFolderId = f.id;
                state.selectedFolderPath = f.destination_path;
                document.querySelector('#selected-folder-display span').textContent = f.destination_path;
                document.getElementById('selected-folder-display').classList.remove('hidden');
                document.getElementById('regex-work-area').style.opacity = '1';
                document.getElementById('regex-work-area').style.pointerEvents = 'auto';
                list.classList.add('hidden');
                input.value = '';

                // Load existing rules
                loadExistingRules(f.destination_path, f.id);
            };
            list.appendChild(item);
        });
    };

    input.addEventListener('input', (e) => renderFolders(e.target.value));
    renderFolders('');
}

async function loadExistingRules(folderPath, folderId) {
    const container = document.getElementById('existing-rules-container');
    if (!container) {
        // Inject container if not exists
        const workArea = document.getElementById('regex-work-area');
        const newContainer = document.createElement('div');
        newContainer.id = 'existing-rules-container';
        newContainer.className = 'mb-6 p-4 rounded border border-border bg-surface';
        newContainer.innerHTML = `
            <div onclick="toggleExistingRules()" class="flex justify-between items-center cursor-pointer select-none">
                <h3 class="font-bold text-lg text-main">Existing Rules</h3>
                <span id="existing-rules-icon">▼</span>
            </div>
            <div id="existing-rules-content" class="hidden mt-4 grid gap-4">
                <div class="text-sm text-muted">Loading...</div>
            </div>
        `;
        // Insert before the "Add New Rule" area/cards or at top
        workArea.insertBefore(newContainer, workArea.firstChild);
    }

    const content = document.getElementById('existing-rules-content');
    content.innerHTML = '<div class="text-sm text-muted">Loading...</div>';

    // Ensure it's closed by default/reset state if desired, or keep user preference
    // User requested "closed by default"
    content.classList.add('hidden');
    document.getElementById('existing-rules-icon').textContent = '▼';

    try {
        const response = await api.getRegexes(folderId);
        // Correctly extract array from response object
        const rules = response.regex_patterns || [];

        // Remove client-side filtering as API already filters by folderId
        // and rules do not contain destination_path
        const folderRules = rules;

        if (folderRules.length === 0) {
            content.innerHTML = '<div class="text-sm text-muted italic">No existing rules for this folder.</div>';
        } else {
            content.innerHTML = '';
            folderRules.forEach(rule => {
                const el = document.createElement('div');
                el.className = 'p-3 rounded bg-base border border-border/50 text-sm';
                el.innerHTML = `
                    <div class="font-mono text-accent mb-1">${rule.regex_pattern}</div>
                    <div class="text-muted">→ ${rule.rename_pattern}</div>
                `;
                content.appendChild(el);
            });
        }
    } catch (e) {
        console.error('Error loading rules:', e);
        content.innerHTML = `<div class="text-danger text-sm">Failed to load rules: ${e.message}</div>`;
    }
}

function toggleExistingRules() {
    const content = document.getElementById('existing-rules-content');
    const icon = document.getElementById('existing-rules-icon');
    if (content.classList.contains('hidden')) {
        content.classList.remove('hidden');
        icon.textContent = '▲';
    } else {
        content.classList.add('hidden');
        icon.textContent = '▼';
    }
}

function clearFolderSelection() {
    state.selectedFolderId = null;
    state.selectedFolderPath = null;
    document.getElementById('selected-folder-display').classList.add('hidden');
    document.getElementById('regex-work-area').style.opacity = '0.5';
    document.getElementById('regex-work-area').style.pointerEvents = 'none';

    // Clear rules
    const container = document.getElementById('existing-rules-container');
    if (container) container.remove();
}

// File Modal
async function openFileModal() {
    const modal = document.getElementById('file-modal');
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    const list = document.getElementById('file-list');
    list.innerHTML = '<div class="text-slate-500 p-4">Loading files...</div>';

    try {
        const files = await api.listFiles('', 'files'); // Root of FILES_ROOT
        // Filter by VIDEO_EXTS (TODO: get from config or hardcode common)
        const vidExts = ['mkv', 'mp4', 'avi', 'm4v', 'mov', 'ts'];
        const vidFiles = files.filter(f => !f.is_dir && vidExts.some(ext => f.name.toLowerCase().endsWith(ext)));

        list.innerHTML = '';
        vidFiles.forEach(f => {
            const row = document.createElement('div');
            row.className = 'file-row';
            row.innerHTML = `
                <input type="checkbox" value="${f.name}" class="input-checkbox">
                <span class="text-main">${f.name}</span>
            `;
            list.appendChild(row);
        });
        // Search Filter
        document.getElementById('file-search').oninput = (e) => {
            const val = e.target.value;
            let regex = null;
            try {
                // Try to create regex from input (case insensitive)
                regex = new RegExp(val, 'i');
            } catch (e) {
                // Invalid regex, regex stays null
            }

            const rows = document.querySelectorAll('.file-row');
            rows.forEach(row => {
                const text = row.querySelector('span').textContent;
                let match = false;
                if (regex) {
                    match = regex.test(text);
                } else {
                    // Fallback to simple includes if regex is empty (matches all) or we decide to treat invalid regex as text search?
                    // User requested "interpret text as Regex". If invalid, maybe don't match, or match nothing?
                    // Let's fallback to includes if basic text, but if it looks like regex but fails?
                    // Safest: if val is empty, match all. If val is invalid regex, maybe just text search?
                    // Actually, "interpret text as Regular Expression" implies if I type ".*mkv", it should work.
                    // If I type "[", it crashes new RegExp.
                    // Let's do: if invalid regex, search as literal text.
                    match = text.toLowerCase().includes(val.toLowerCase());
                }
                row.style.display = match ? 'flex' : 'none';
            });
        };
    } catch (e) {
        list.innerHTML = `<div class="text-accent p-4">${e.message}</div>`;
    }
}

function closeFileModal() {
    document.getElementById('file-modal').classList.add('hidden');
    document.getElementById('file-modal').classList.remove('flex');
}

function addSelectedFiles() {
    const checks = document.querySelectorAll('#file-list input:checked');
    checks.forEach(c => {
        const filename = c.value;
        addRegexCard(filename);
    });
    closeFileModal();
}

function addRegexCard(filename) {
    const id = Date.now() + Math.random().toString(36).substr(2, 9);
    const container = document.getElementById('regex-cards');

    // Pre-fill logic
    // Escape all regex special characters: \ ^ $ * + ? . ( ) | { } [ ]
    // Using a robust escape function
    const escapeRegex = (string) => {
        return string.replace(/[\*\+\?\^\$\{\}\(\)\|\[\]\\]/g, '\\$&');
    };

    const escapedName = escapeRegex(filename);
    // Regex Pattern: Pre-fill with filename (escaped)
    const regexPattern = escapedName;
    // Rename Pattern: Pre-fill with filename (original)
    const renamePattern = filename;

    const card = document.createElement('div');
    card.id = `card-${id}`;
    card.className = 'card mt-4';
    card.innerHTML = `
        <div style="display:flex; justify-content:space-between; margin-bottom: 1rem;">
             <span class="text-muted text-xs uppercase font-bold">Target File</span>
             <button onclick="removeCard('${id}')" class="btn-icon">🗑️</button>
        </div>
        <div class="font-mono text-sm text-accent p-2 mb-4" style="background:var(--bg-surface); border-radius:4px; word-break: break-all;" title="${filename}">${filename}</div>
        
        <div class="grid-2">
            <div>
                <label class="label">Regex Pattern</label>
                <input type="text" value="${regexPattern}" oninput="validateCard('${id}', '${filename}')" class="regex-input input font-mono" placeholder="e.g. S\\d{2}E(?P<episode>\\d+)">
                <div class="mt-2 text-xs text-muted validation-msg"></div>
            </div>
            <div>
                <label class="label">Rename Pattern</label>
                <input type="text" value="${renamePattern}" oninput="validateCard('${id}', '${filename}')" class="rename-input input font-mono" placeholder="e.g. MyShow - {<episode>} - Title">
                <div class="mt-2 text-xs text-muted rename-msg"></div>
            </div>
        </div>
        <div class="mt-4 flex items-center gap-4">
             <div>
                <label class="label">Weight</label>
                <input type="number" value="1" class="weight-input input text-center" style="width: 5rem;">
            </div>
            
            <!-- Extraction Display -->
            <div class="flex-1 flex flex-col items-center justify-center p-4 rounded border border-accent/30" style="background:var(--bg-surface);">
                <span class="text-xs text-muted uppercase tracking-wider mb-1">Extracted Episode</span>
                <span class="text-3xl font-bold text-success extracted-ep">-</span>
            </div>
        </div>

        <!-- Card Footer -->
        <div class="mt-4 pt-4 border-t border-border flex justify-end gap-2">
            <button onclick="submitSingleRule('${id}', true)" class="btn btn-secondary btn-sm">Save & Close</button>
            <button onclick="submitSingleRule('${id}', false)" class="btn btn-primary btn-sm">Save Rule</button>
        </div>
    `;

    container.appendChild(card);
    state.regexCards.push({ id, filename, el: card });
    // Run validation initially since we pre-filled
    setTimeout(() => validateCard(id, filename), 0);
}

function removeCard(id) {
    const card = document.getElementById(`card-${id}`);
    card.remove();
    state.regexCards = state.regexCards.filter(c => c.id !== id);
}

function validateCard(id, filename) {
    const card = document.getElementById(`card-${id}`);
    const input = card.querySelector('.regex-input').value;
    const msg = card.querySelector('.validation-msg');
    const epDisplay = card.querySelector('.extracted-ep');

    const renameInput = card.querySelector('.rename-input').value;
    const renameMsg = card.querySelector('.rename-msg');

    // Regex Validation
    let regexValid = false;
    try {
        // Rust Regex to JS Regex conversion basics
        // Rust: (?P<name>...) -> JS: (?<name>...)
        let jsRegexStr = input.replace(/\?P<([\w]+)>/g, '?<$1>');
        const re = new RegExp(jsRegexStr);
        const match = filename.match(re);

        if (match && match.groups && match.groups.episode) {
            msg.textContent = "✓ Valid Pattern";
            msg.className = "mt-2 text-xs text-success validation-msg";
            epDisplay.textContent = match.groups.episode;
            regexValid = true;
        } else {
            msg.textContent = "⚠ No 'episode' group match";
            msg.className = "mt-2 text-xs text-warning validation-msg";
            epDisplay.textContent = "-";
        }
    } catch (e) {
        msg.textContent = "⚠ Invalid Regex Syntax";
        msg.className = "mt-2 text-xs text-danger validation-msg";
    }

    // Rename Pattern Validation
    if (renameInput && !renameInput.includes('{<episode>}')) {
        renameMsg.textContent = "⚠ Must contain {<episode>}";
        renameMsg.className = "mt-2 text-xs text-warning rename-msg";
    } else {
        renameMsg.textContent = "";
        renameMsg.className = "mt-2 text-xs text-muted rename-msg";
    }
}

async function submitSingleRule(id, closeOnSuccess = false) {
    const item = state.regexCards.find(c => c.id === id);
    if (!item) return;
    const card = item.el;

    const regexVal = card.querySelector('.regex-input').value;
    const renameVal = card.querySelector('.rename-input').value;
    const weight = parseInt(card.querySelector('.weight-input').value, 10);

    // Validate again just in case
    if (!regexVal) {
        showToast('Regex pattern is required', 'error');
        return;
    }

    const payload = {
        regex_pattern: regexVal,
        rename_pattern: renameVal,
        weight: weight,
        example_filename: item.filename,
        destination_path: state.selectedFolderPath
    };

    try {
        await api.createRegex(payload);
        showToast('Rule saved successfully!', 'success');

        if (closeOnSuccess) {
            removeCard(id);
        } else {
            // Optional: Visual feedback like disabling button or changing text
            const btn = card.querySelector('button.btn-primary');
            if (btn) {
                btn.textContent = 'Saved ✓';
                btn.classList.remove('btn-primary');
                btn.classList.add('btn-secondary'); // or checkmark style
                setTimeout(() => {
                    btn.textContent = 'Save Rule';
                    btn.classList.add('btn-primary');
                    btn.classList.remove('btn-secondary');
                }, 3000);
            }
        }

    } catch (e) {
        showToast('Error saving rule: ' + e.message, 'error');
    }
}

// submitAllRegex removed in favor of single rule submission

// Init
// initRegexTab(); // Removed auto-init, handled by switchTab
